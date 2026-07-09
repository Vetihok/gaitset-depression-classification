import logging
import math
import os
import os.path as osp
import random
import re
import sys
from copy import deepcopy
from datetime import datetime

import numpy as np
import json
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import torch.nn.functional as F
from progress.bar import Bar
from sklearn.metrics import f1_score, recall_score, confusion_matrix

from config import conf
from opts import get_opts
from .network import BinaryClassificationNet, CETripletLoss
from .utils import ClassificationSampler, CETripletSampler
from classify_depression import get_cwd, get_general_save_path
from metrics_utils import plot_metric_curve2

log = logging.getLogger(__name__)

def get_shape(obj):
    """Return the shape (dimensions) of a nested list or tuple as a tuple."""
    if not isinstance(obj, (list, tuple)):
        return ()  # scalars have empty shape
    return (len(obj),) + get_shape(obj[0]) if obj else (0,)


def resolve_device(use_cpu=False):
    """Return the torch device to use for tensors and modules."""
    if use_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_state_dict_for_load(state_dict, use_cpu=False, strip_prefixes=None, model=None, add_module_prefix=True):
    """Normalize checkpoint keys to match the current model architecture.

    The checkpoints can be saved either from a plain module or from a
    DataParallel-wrapped module. This helper removes or adds the ``module.``
    prefix when needed, and can also strip a leading submodule prefix such as
    ``encoder.`` when loading a submodule state dict.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    normalized = {}
    prefixes = list(strip_prefixes or [])
    expects_module_prefix = isinstance(model, nn.DataParallel) if model is not None else (not use_cpu)

    for key, value in state_dict.items():
        normalized_key = key
        for prefix in prefixes:
            if normalized_key == prefix:
                normalized_key = ""
            elif normalized_key.startswith(prefix + "."):
                normalized_key = normalized_key[len(prefix) + 1:]

        if add_module_prefix and expects_module_prefix:
            if not normalized_key.startswith("module."):
                normalized_key = f"module.{normalized_key}"
        elif not add_module_prefix:
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module."):]
        else:
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module."):]

        normalized[normalized_key] = value

    return normalized


class Model:
    def __init__(self,
                 hidden_dim,
                 num_workers,
                 batch_size,
                 restore_epoch,
                 eval_interval,
                 num_epochs,
                 opt_cfg,
                 loss_cfg,
                 sch_cfg,
                 rdrop_cfg,
                 dropout_cfg,
                 sampler_cfg,
                 early_stop_cfg,
                 freeze_cfg,
                 classifier_head_cfg,
                 save_name,
                 frame_num,
                 model_name,
                 train_source,
                 val_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        #self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        #self.lr = lr
        self.frame_num = frame_num
        self.num_workers = num_workers
        
        self.batch_sample = batch_size
        # Handle batch_size as tuple or int
        if isinstance(batch_size, tuple):
            self.batch_size = batch_size[0] * batch_size[1]
        else:
            self.batch_size = batch_size
        
        self.model_name = model_name

        #self.restore_iter = restore_iter
        self.restore_epoch = restore_epoch
        self.epoch = 1 if self.restore_epoch == 0 else self.restore_epoch
        #self.total_iter = total_iter
        self.num_epochs = num_epochs
        self.eval_interval = eval_interval
        self.img_size = img_size

        # Initialize binary classification network
        self.device = resolve_device(conf.get("USE_CPU", False))
        self.model = BinaryClassificationNet(
            dropout_cfg,
            freeze_cfg,
            self.hidden_dim,
            classifier_head_cfg=classifier_head_cfg,
        ).float()
        if not conf.get("USE_CPU", False):
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # --- Loss ---
        self.ce_loss_cfg = loss_cfg.get("cross_entropy", {})
        self.triplet_cfg = loss_cfg.get("triplet", {})
        self.focal_loss_cfg = loss_cfg.get("focal", {})

        self.criterion = CETripletLoss(self.batch_sample, ce_loss_cfg=self.ce_loss_cfg, triplet_cfg=self.triplet_cfg, focal_loss_cfg=self.focal_loss_cfg)
        self.triplet_cfg['enabled'] = False
        self.val_criterion = CETripletLoss(self.batch_sample, ce_loss_cfg=self.ce_loss_cfg, triplet_cfg=self.triplet_cfg, focal_loss_cfg=self.focal_loss_cfg)
        self.triplet_cfg['enabled'] = True
        self.criterion.to(self.device)
        self.val_criterion.to(self.device)
        
        # --- R-Drop ---
        self.rdrop_enabled = rdrop_cfg.get("enabled", False)
        self.rdrop_alpha = rdrop_cfg.get("alpha", 1.0)
        
        if self.rdrop_enabled and not dropout_cfg.get("enabled", False):
            log.warning("R-Drop is activated but dropout is disactivated : R-Drop will have no effect.")

        # --- Optimizer ---
        opt_type = opt_cfg.get("type", "Adam")
        self.lr = opt_cfg.get("lr", 0.0001)
        wd = opt_cfg.get("weight_decay", 0.0)

        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown Optimizer : {opt_type}")

        # --- Scheduler ---
        sch_type = sch_cfg.get("type", "CosineAnnealingLR")
        self.sch_enabled = sch_cfg.get("enabled", True)

        if self.sch_enabled:
            if sch_type == "CosineAnnealingLR":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                    T_max=self.num_epochs,
                    eta_min=sch_cfg.get("eta_min", 0),
                )
            elif sch_type == "OneCycleLR":
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                    max_lr=sch_cfg.get("max_lr", self.lr * 10),
                    total_steps=self.num_epochs,
                    pct_start=sch_cfg.get("pct_start", 0.3),
                )
            elif sch_type == "PolynomialLR":
                self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer,
                    total_iters=self.num_epochs,
                    power=sch_cfg.get("power", 1.0),
                )
            elif sch_type == "MultiStepLR":
                self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=sch_cfg.get("milestones", [10000, 20000, 30000]),
                    gamma=sch_cfg.get("gamma", 0.1),
                )
            else:
                raise ValueError(f"Unknown scheduler : {sch_type}")

        # --- Sampler ---
        self.sampler_type = sampler_cfg.get("type", "ClassificationSampeler")
        self.val_batch_sampler = tordata.BatchSampler(
                tordata.sampler.SequentialSampler(self.val_source),
                batch_size=self.batch_size,
                drop_last=False
            )
        if self.sampler_type == "WeightedRandomSampler":
            classes, counts = np.unique(self.train_source.label, return_counts=True)
            class_weights = 1. /(counts * sampler_cfg.get("weight_damping_factor", 1.0))
            sample_weights = class_weights[self.train_source.label]
            log.debug(f"class_weights = {class_weights}")
            log.debug(f"Label distribution: {np.unique(self.train_source.label, return_counts=True)}")
            log.debug(f"Total samples: {len(self.train_source.label)}, Batch size: {self.batch_size}")

            train_sampler = tordata.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.train_source), # sample from the dataset size
                replacement=True # allows sampling the same index multiple times
            )
            self.train_batch_sampler = tordata.BatchSampler(
                train_sampler,
                batch_size=self.batch_size,
                drop_last=False
            )
        elif self.sampler_type == "ClassificationSampler":
            self.train_batch_sampler = ClassificationSampler(self.train_source, self.batch_size)
        elif self.sampler_type == "SequentialSampler":
            self.train_batch_sampler = tordata.BatchSampler(
                tordata.sampler.SequentialSampler(self.train_source),
                batch_size=self.batch_size,
                drop_last=False
            )
        elif self.sampler_type == "RandomSampler":
            self.train_batch_sampler = tordata.BatchSampler(
                tordata.sampler.RandomSampler(self.train_source),
                batch_size=self.batch_size,
                drop_last=False
            )
        elif self.sampler_type == "CETripletSampler":
            self.train_batch_sampler = CETripletSampler(self.train_source, self.batch_sample)
        else:
            raise ValueError(f"Unknown sampler : {sch_type}")

        self.early_stop = 0 if not early_stop_cfg.get("enabled", True) else early_stop_cfg.get("patience", True)

        # --- freeze ---
        self.unfreeze_epoch = freeze_cfg.get("unfreeze_epoch", -1) if freeze_cfg.get('enabled', False) else -1
        self.invert_freezing_epoch = freeze_cfg.get("invert_freezing_epoch", -1) if freeze_cfg.get('enabled', False) else -1
        if freeze_cfg.get('freeze_whole_encoder', False):
            log.warning("Whole encoder is frozen.")

        self.history = None
        self.sample_type = sampler_cfg.get("sample_type", "all")

        aug_cfg = conf.get("model", {}).get("augmentation", {})
        self.augment_enabled = aug_cfg.get("enabled", False)
        self.augment_prob = aug_cfg.get("prob", 0.5)
        self.augment_horizontal_flip = aug_cfg.get("horizontal_flip", True)
        self.augment_gaussian_noise = aug_cfg.get("gaussian_noise", True)
        self.augment_random_erasing = aug_cfg.get("random_erasing", True)
        # Additional augmentations
        self.augment_random_translation = aug_cfg.get("random_translation", True)
        self.augment_max_translation = aug_cfg.get("max_translation", 4)  # pixels
        self.augment_rotation = aug_cfg.get("rotation", True)
        self.augment_max_rotation = aug_cfg.get("max_rotation", 5.0)  # degrees
        self.augment_border_erasing = aug_cfg.get("border_erasing", True)
        self.augment_border_erasing_prob = aug_cfg.get("border_erasing_prob", 0.5)
        self.augment_gaussian_blur = aug_cfg.get("gaussian_blur", True)
        self.augment_blur_sigma = aug_cfg.get("blur_sigma", 0.8)
        self.augment_border_scale = aug_cfg.get("border_scale", True)
        self.augment_border_scale_px = aug_cfg.get("border_scale_px", 1)  # pixels to dilate/erode

    def _augment_sequence(self, seq_array):
        if not self.augment_enabled or not self.model.training:
            return seq_array

        if not isinstance(seq_array, np.ndarray):
            return seq_array
        if seq_array.ndim != 3:
            return seq_array

        aug = seq_array.copy()
        # If we are not applying augmentation this time, keep original
        if random.random() >= self.augment_prob:
            return aug

        # Build list of activated augmentations and choose one
        possible_augs = []
        if self.augment_horizontal_flip:
            possible_augs.append('flip')
        if self.augment_gaussian_noise:
            possible_augs.append('noise')
        if self.augment_random_translation:
            possible_augs.append('translation')
        if self.augment_rotation:
            possible_augs.append('rotation')
        if self.augment_random_erasing:
            possible_augs.append('random_erasing')
        if self.augment_border_erasing:
            possible_augs.append('border_erasing')
        if self.augment_gaussian_blur:
            possible_augs.append('gaussian_blur')
        if self.augment_border_scale:
            possible_augs.append('border_scale')

        if not possible_augs:
            return aug

        choice = random.choice(possible_augs)

        if choice == 'flip':
            aug = np.flip(aug, axis=-1)

        elif choice == 'noise':
            noise = np.random.normal(0.0, 0.01, size=aug.shape).astype(np.float32)
            aug = np.clip(aug + noise, 0.0, 1.0)

        elif choice == 'translation':
            tx = random.randint(-self.augment_max_translation, self.augment_max_translation)
            ty = random.randint(-self.augment_max_translation, self.augment_max_translation)
            if tx != 0 or ty != 0:
                a = np.roll(aug, shift=tx, axis=-2) if tx else aug
                a = np.roll(a, shift=ty, axis=-1) if ty else a
                if tx > 0:
                    a[..., :tx, :] = 0
                elif tx < 0:
                    a[..., tx:, :] = 0
                if ty > 0:
                    a[..., :, :ty] = 0
                elif ty < 0:
                    a[..., :, ty:] = 0
                aug = a

        elif choice == 'rotation':
            rot_angle = random.uniform(-self.augment_max_rotation, self.augment_max_rotation)
            try:
                from scipy.ndimage import rotate as _rotate
                rr = np.empty_like(aug)
                for i in range(aug.shape[0]):
                    rr[i] = _rotate(aug[i], rot_angle, reshape=False, order=1, mode='constant', cval=0.0)
                aug = rr
            except Exception:
                pass

        elif choice == 'random_erasing':
            h, w = aug.shape[-2], aug.shape[-1]
            if h > 8 and w > 8:
                erase_h = max(4, int(h * random.uniform(0.08, 0.2)))
                erase_w = max(4, int(w * random.uniform(0.08, 0.2)))
                top = random.randint(0, h - erase_h)
                left = random.randint(0, w - erase_w)
                aug[..., top:top + erase_h, left:left + erase_w] = 0.0

        elif choice == 'border_erasing':
            A = aug.copy()
            def erode_mask(mask):
                m = mask.astype(bool)
                up = np.roll(m, -1, axis=-2)
                down = np.roll(m, 1, axis=-2)
                left = np.roll(m, -1, axis=-1)
                right = np.roll(m, 1, axis=-1)
                up[..., -1:, :] = False
                down[..., :1, :] = False
                left[..., :, -1:] = False
                right[..., :, :1] = False
                return m & up & down & left & right
            mask = (A > 0.5)
            eroded = erode_mask(mask)
            edge = mask & (~eroded)
            drop = (np.random.rand(*edge.shape) < 0.5) & edge
            A[drop] = 0.0
            aug = A

        elif choice == 'gaussian_blur':
            try:
                from scipy.ndimage import gaussian_filter
                B = np.empty_like(aug)
                for i in range(aug.shape[0]):
                    B[i] = gaussian_filter(aug[i], sigma=self.augment_blur_sigma)
                aug = B
            except Exception:
                pad = ((0, 0), (1, 1), (1, 1))
                p = np.pad(aug, pad, mode='constant', constant_values=0)
                B = np.empty_like(aug)
                for i in range(aug.shape[0]):
                    f = p[i]
                    s = (f[:-2, :-2] + f[:-2, 1:-1] + f[:-2, 2:] +
                         f[1:-1, :-2] + f[1:-1, 1:-1] + f[1:-1, 2:] +
                         f[2:, :-2] + f[2:, 1:-1] + f[2:, 2:]) / 9.0
                    B[i] = s
                aug = B

        elif choice == 'border_scale':
            S = aug.copy()
            px = max(1, int(self.augment_border_scale_px))
            for _ in range(px):
                m = (S > 0.5)
                up = np.roll(m, -1, axis=-2)
                down = np.roll(m, 1, axis=-2)
                left = np.roll(m, -1, axis=-1)
                right = np.roll(m, 1, axis=-1)
                up[..., -1:, :] = False
                down[..., :1, :] = False
                left[..., :, -1:] = False
                right[..., :, :1] = False
                S = (m | up | down | left | right).astype(np.float32)
            if random.random() < 0.5:
                for _ in range(px):
                    m = (S > 0.5)
                    up = np.roll(m, -1, axis=-2)
                    down = np.roll(m, 1, axis=-2)
                    left = np.roll(m, -1, axis=-1)
                    right = np.roll(m, 1, axis=-1)
                    up[..., -1:, :] = False
                    down[..., :1, :] = False
                    left[..., :, -1:] = False
                    right[..., :, :1] = False
                    S = (m & up & down & left & right).astype(np.float32)
            aug = S

        return aug

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        patient_id = [batch[i][5] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None, patient_id]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.model.training and self.augment_enabled:
            for i in range(batch_size):
                for j in range(feature_num):
                    seqs[i][j] = self._augment_sequence(seqs[i][j])

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    @staticmethod
    def compute_rdrop_loss(criterion, logits1, logits2, target_label, alpha):
        """
        R-Drop loss: NLL sur les deux passes + divergence KL bidirectionnelle.
        Loss = -logP1 - logP2 + (alpha/2) * (KL(P1||P2) + KL(P2||P1))
        """
        loss_nll = criterion(logits1, target_label) + criterion(logits2, target_label)

        p1 = torch.softmax(logits1, dim=-1)
        p2 = torch.softmax(logits2, dim=-1)

        # KL(P1||P2) + KL(P2||P1)  — F.kl_div attend log-probs en entrée
        loss_kl = (
            F.kl_div(p2.log(), p1, reduction='batchmean') +
            F.kl_div(p1.log(), p2, reduction='batchmean')
        )

        return loss_nll + (alpha / 2.0) * loss_kl

    def fit(self):
        """Train the binary classification model"""

        self.model.train()
        self.sample_type = 'random'
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=self.train_batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers
        )

        val_loader = tordata.DataLoader(
            dataset=self.val_source,
            batch_sampler=self.val_batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            #pin_memory=False,  # Désactive le pinning pour éviter la saturation mémoire
        )

        train_pid_set = list(set(self.train_source.patient_id))
        train_pid_set.sort()
        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()
        val_pid_set = list(set(self.val_source.patient_id))
        val_pid_set.sort()
        val_label_set = list(self.val_source.label_set)
        val_label_set.sort()

        #log.debug(f"{str(train_label_set)=:.100}")
        #log.debug(f"{len(train_label_set)=}")
        #log.debug(f"{str(train_label_set)=:.100}")

        best_val_f1 = -1
        self.best_val_f1_epoch = 0
        eval_interval = self.eval_interval
        training_iterations = len(train_loader)
        log.debug(f"{training_iterations=}")

        loss_history = []
        acc_history = []
        f1_history = []
        recall_history = []

        avg_loss_history = []
        avg_acc_history = []
        avg_f1_history = []
        avg_recall_history = []

        val_loss_history = []
        val_acc_history = []
        val_f1_history = []
        val_recall_history = []
        val_iterations = []  # Track iteration numbers for each validation measurement


        if eval_interval >= self.num_epochs:
            log.warning(f"No evaluation during training : {eval_interval=} >= {self.num_epochs=}")

        is_unfrozen = False
        log.debug(f"{len(val_loader.dataset)=}, {len(train_loader.dataset)=}, {len(self.val_source)=}, {len(self.train_source)=}, {len(self.test_source)=}")

        while self.epoch <= self.num_epochs:

            if self.epoch == self.unfreeze_epoch and not is_unfrozen:
                log.info(f"[Époque {self.epoch}] Dégel complet de l'encodeur pour le fine-tuning !")
                self.model.module.unfreeze_encoder()
                is_unfrozen = True
            
            if self.epoch == self.invert_freezing_epoch and not is_unfrozen:
                log.info(f"Unfreezing encoder and freezing classifier at epoch {self.epoch}")
                self.model.module.unfreeze_encoder()
                self.model.module.freeze_classifier()
                is_unfrozen = True

            iter = 0
            tn_tr, fp_tr, fn_tr, tp_tr = 0, 0, 0, 0

            bar = Bar(f"Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}", max=len(train_loader.dataset))
            for seq, view, seq_type, label, batch_frame, patient_id in train_loader:
                self.optimizer.zero_grad()
                
                # Convert sequences to tensors
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                # Get target labels (0=Normal, 1=Depressed)
                target_label = [train_label_set.index(l) for l in label]
                log.debug(f"{str(target_label)=}")
                target_label = self.np2var(np.array(target_label)).long()

                # --- before (one forward pass) ---
                logits, features = self.model(*seq, batch_frame)

                # log.debug(f"{len(*seq)=}")
                log.debug(f"{len(label)=}, {label=}")
                # classes, counts = np.unique(label, return_counts=True)
                # log.debug(f"{counts=}")
                # log.debug(f"{len(target_label)=}, {target_label=}")
                log.debug(f"{len(patient_id)=}, {patient_id=}")
                # log.debug(f"{len(logits)=}, {logits=}")
                # log.debug(f"{len(features)=}, {features=}")

                target_pid = [train_pid_set.index(l) for l in patient_id]
                log.debug(f"{str(target_pid)=}") 
                target_pid = self.np2var(np.array(target_pid)).long()

                if self.triplet_cfg.get('enabled', False):
                    loss = self.criterion(logits, features, target_label, target_pid)
                else:
                    loss = self.criterion(logits, target_label)

                """# Afficher les shapes pour TOUTES les modalités
                for idx, s in enumerate(seq):
                    log.debug(f"Input[{idx}] shape: {s.shape}, dtype: {s.dtype}, device: {s.device}")

                # Afficher les valeurs brutes du PREMIER sample (index 0) pour chaque modalité
                for idx, s in enumerate(seq):
                    for i in range(3):
                        sample_data = s[i]  # Premier sample du batch
                        log.debug(f"Sample {i}, Modalité {idx}: shape={sample_data.shape}, min={sample_data.min():.3f}, max={sample_data.max():.3f}, mean={sample_data.mean():.3f}")

                        # Afficher les métadonnées du premier sample
                        if len(label) > i:
                            log.debug(f"Sample {i} metadata: view={view[i]}, seq_type={seq_type[i]}, label={label[i]}, batch_frame={batch_frame[i] if batch_frame is not None else 'None'}")
                """
                

                # --- after (conditionnal R-Drop) ---
                if self.rdrop_enabled:
                    # logits, features   = self.model(*seq, batch_frame)   # passe 1
                    logits2, _         = self.model(*seq, batch_frame)   # passe 2 (dropout différent)
                    loss = self.compute_rdrop_loss(
                        self.criterion, logits, logits2, target_label, self.rdrop_alpha
                    )
                # else:
                #     logits, features = self.model(*seq, batch_frame)
                #     loss = self.criterion(logits, target_label)

                # Backward pass
                if loss > 1e-9:
                    loss.backward()
                    self.optimizer.step()

                # Track metrics
                # with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = (pred == target_label).float().mean().item()

                pred_np = pred.cpu().numpy()
                target_np = target_label.cpu().numpy()
                
                f1 = f1_score(target_np, pred_np, average='binary', zero_division=0)
                recall = recall_score(target_np, pred_np, average='binary', zero_division=0)
                tn, fp, fn, tp = confusion_matrix(target_np, pred_np, labels=[0, 1]).ravel()
                tn_tr += tn
                fp_tr += fp
                fn_tr += fn
                tp_tr += tp

                loss_history.append(loss.item())
                acc_history.append(acc)
                f1_history.append(f1)
                recall_history.append(recall)

                # iter += en(label)
                bar.next(len(label))
            
            bar.finish()
            
            if self.sch_enabled:
                self.scheduler.step()
            # Print stats
            avg_loss = np.mean(loss_history[-training_iterations:]) if len(loss_history) >= training_iterations else np.mean(loss_history)
            avg_acc = np.mean(acc_history[-training_iterations:]) if len(acc_history) >= training_iterations else np.mean(acc_history)
            avg_f1 = np.mean(f1_history[-training_iterations:]) if len(f1_history) >= training_iterations else np.mean(f1_history)
            avg_recall = np.mean(recall_history[-training_iterations:]) if len(recall_history) >= training_iterations else np.mean(recall_history)

            avg_loss_history.append(avg_loss)
            avg_acc_history.append(avg_acc)
            avg_f1_history.append(avg_f1)
            avg_recall_history.append(avg_recall)

            log.info(f'Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}: loss={avg_loss:.8f}, acc={avg_acc:.8f}, f1={avg_f1:.8f}, recall={avg_recall:.8f}, '
                        f'lr={self.optimizer.param_groups[0]["lr"]:.6f}')
            log.info(f"Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}: TP={tp_tr}, FP={fp_tr}, TN={tn_tr}, FN={fn_tr}")

            if self.epoch % eval_interval == 0:
                # Evaluate on test set
                self.model.eval()
                # val_loss = 0.0
                # val_acc = 0.0
                # val_f1 = 0.0
                # val_recall = 0.0
                # tn_val, fp_val, fn_val, tp_val = 0, 0, 0, 0

                # try:
                #     # with torch.no_grad():
                #     outputs = self.transform('val', batch_size=1) #self.batch_size//16)
                #     val_logits = outputs['logits']
                #     val_labels = outputs['labels']
                #     val_features = outputs['features']
                #     val_pid = outputs['patient_ids']
                #     # val_prob = outputs['probabilities']
                #     if len(val_labels) > 0:
                #         val_logits_t = torch.from_numpy(val_logits).cuda()
                #         val_features_t = torch.from_numpy(val_features).cuda()
                #         #val_prob_t = torch.from_numpy(val_prob).cuda()

                #         target_val = np.array([val_label_set.index(l) for l in val_labels])
                #         target_val_t = torch.from_numpy(target_val).long().cuda()

                #         target_pid = np.array([val_pid_set.index(l) for l in val_pid])
                #         target_pid_t = torch.from_numpy(target_pid).long().cuda()
                #         if self.triplet_cfg.get('enabled', False):
                #             val_loss = self.criterion(val_logits_t, val_features_t, target_val_t, target_pid_t).item()
                #         else:
                #             val_loss = self.criterion(val_logits_t, target_val_t).item()
                        
                #         val_pred = torch.argmax(val_logits_t, dim=1)
                #         val_acc = (val_pred == target_val_t).float().mean().item()

                #         # Calcul F1 et Recall pour classification binaire/multi-classe
                #         val_pred_np = val_pred.cpu().numpy()
                #         target_val_np = target_val_t.cpu().numpy()
                #         val_f1 = f1_score(target_val_np, val_pred_np, average='binary')
                #         val_recall = recall_score(target_val_np, val_pred_np, average='binary')
                #         tn_val, fp_val, fn_val, tp_val = confusion_matrix(target_val_np, val_pred_np, labels=[0, 1]).ravel()
                #     else:
                #         log.warning(f'No validation labels found at epoch {self.epoch}')
                # except Exception as e:
                #     log.error(f'Error during validation at epoch {self.epoch}: {e}')
                #     import traceback
                #     log.error(traceback.format_exc())
                #     val_loss = 0.0
                #     val_acc = 0.0
                #     val_f1 = 0.0
                #     val_recall = 0.0

                # # Record validation metrics
                # val_loss_history.append(val_loss)
                # val_acc_history.append(val_acc)
                # val_f1_history.append(val_f1)
                # val_recall_history.append(val_recall)
                # val_iterations.append(self.epoch)
                total_loss = 0.0
                all_logits = []
                all_labels = []
                all_pids = []

                with torch.no_grad():
                    bar = Bar(f"Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}", max=len(val_loader.dataset))
                    for seq, view, seq_type, label, batch_frame, patient_id in val_loader:
                        # Convertir en tensors GPU
                        for i in range(len(seq)):
                            seq[i] = self.np2var(seq[i]).float()
                        if batch_frame is not None:
                            batch_frame = self.np2var(batch_frame).int()

                        # Calculer la perte SUR CE BATCH (sans accumuler toutes les features)
                        batch_label = [val_label_set.index(l) for l in label]
                        batch_label_t = self.np2var(np.array(batch_label)).long()

                        # Forward pass
                        logits, features = self.model(*seq, batch_frame)
                        # log.debug(f"{get_shape(seq)=}")
                        # log.debug(f"{type(logits)=}, {logits.data.cpu().numpy().shape=}")
                        # log.debug(f"{type(features)=}, {features.data.cpu().numpy().shape=}")
                        # log.debug(f"{len(label)=}, {get_shape(label)=}, {label=}")
                        # log.debug(f"{len(batch_label_t)=}, {batch_label_t.shape=}, {batch_label_t=}")
                        # log.debug(f"{len(patient_id)=}, {get_shape(patient_id)=}, {patient_id=}")
                        # Stocker les logits/labels pour les métriques (sur CPU)
                        all_logits.append(logits.cpu())
                        all_labels.extend(label)
                        all_pids.extend(patient_id)

                        

                        if self.triplet_cfg.get('enabled', False):
                            batch_loss = self.val_criterion(logits, None, batch_label_t, None).item()
                        else:
                            batch_loss = self.criterion(logits, batch_label_t).item()

                        total_loss += batch_loss * len(label)  # Pondérer par la taille du batch
                        bar.next(len(label))

                    bar.finish()

                val_loss = total_loss / len(val_loader.dataset)

                val_logits = torch.cat(all_logits, dim=0)
                val_pred = torch.argmax(val_logits, dim=1)
                val_pred_np = val_pred.numpy()
                target_val_np = np.array([val_label_set.index(l) for l in all_labels])

                val_acc = (val_pred == torch.tensor(target_val_np)).float().mean().item()
                val_f1 = f1_score(target_val_np, val_pred_np, average='binary', zero_division=0)
                val_recall = recall_score(target_val_np, val_pred_np, average='binary', zero_division=0)
                tn_val, fp_val, fn_val, tp_val = confusion_matrix(target_val_np, val_pred_np, labels=[0, 1]).ravel()

                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                val_f1_history.append(val_f1)
                val_recall_history.append(val_recall)
                val_iterations.append(self.epoch)

        
                log.info(f'Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}: '
                        f'{val_loss=:.8f}, {val_acc=:.8f}, {val_f1=:.8f}, {val_recall=:.8f}, '
                        f'lr={self.optimizer.param_groups[0]["lr"]:.6f}, {best_val_f1=:.6f}')
                log.info(f"Epoch {str(self.epoch).zfill(len(str(self.num_epochs)))}/{self.num_epochs}: TP={tp_val}, FP={fp_val}, TN={tn_val}, FN={fn_val}")
                
                self.history = {
                    'loss': loss_history,
                    'acc': acc_history,
                    'f1': f1_history,
                    'recall': recall_history,
                    'val_loss': val_loss_history,
                    'val_acc': val_acc_history,
                    'val_f1': val_f1_history,
                    'val_recall': val_recall_history,
                    'val_iterations': val_iterations
                }

                plot_metric_curve2(get_general_save_path('loss', 'png'), avg_loss_history, val_loss_history, val_iterations,
                                metric_name="Loss", eval_interval=eval_interval)
                plot_metric_curve2(get_general_save_path('accuracy', 'png'), avg_acc_history, val_acc_history, val_iterations,
                                metric_name="Accuracy", eval_interval=eval_interval)
                plot_metric_curve2(get_general_save_path('f1_score', 'png'), avg_f1_history, val_f1_history, val_iterations,
                    metric_name="F1-Score", eval_interval=eval_interval)
                plot_metric_curve2(get_general_save_path('recall', 'png'), avg_recall_history, val_recall_history, val_iterations,
                    metric_name="Recall", eval_interval=eval_interval)

                # Save only if validation f1-score improved
                if val_f1 > best_val_f1:
                    self.save()
                    log.warning(f"Model saved at epoch {self.epoch}: {val_f1=:.6f}, {best_val_f1=:.6f}")
                    best_val_f1 = val_f1
                    self.best_val_f1_epoch = self.epoch
                
                if self.early_stop > 0 and self.epoch - self.best_val_f1_epoch == self.early_stop:
                    log.warning(f"Early stop : training interupted at epoch {self.epoch}: {best_val_f1=}, epoch={self.best_val_f1_epoch}")
                    break

                # Return to train mode
                self.model.train()

                self.epoch += 1
        
        self.save()
        log.info("Model saved. Training completed!")
        return self.best_val_f1_epoch

    def ts2var(self, x):
        if isinstance(x, torch.Tensor):
            tensor = x
        else:
            tensor = torch.as_tensor(x)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return autograd.Variable(tensor).to(self.device)

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        """Extract features and predictions from the model"""
        self.model.eval()
        source = self.test_source if flag == 'test' else self.train_source
        match flag:
            case 'train':
                source = self.train_source
            case 'val':
                source = self.val_source
            case 'test':
                source = self.test_source
            case _:
                raise ValueError(f'Unknown transform flag : {flag}')
        self.sample_type = 'all'
        
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        logit_list = list()
        prob_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()
        patient_id_list = list()

        with torch.no_grad():
            bar = Bar(f"{flag.capitalize()} transform", max=len(source))
            for seq, view, seq_type, label, batch_frame, patient_id in data_loader:
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                logits, features = self.model(*seq, batch_frame)
                probs = torch.softmax(logits, dim=1)
                
                # features shape: [batch_size, num_bins, hidden_dim]
                # Flatten to [batch_size, num_bins * hidden_dim]
                batch_size, num_bins, hidden_dim = features.shape
                # features_flat = features.contiguous().view(batch_size, -1)
                
                # feature_list.append(features_flat.data.cpu().numpy())
                feature_list.append(features.data.cpu().numpy())
                logit_list.append(logits.data.cpu().numpy())
                prob_list.append(probs.data.cpu().numpy())
                
                view_list += view
                seq_type_list += seq_type
                label_list += label
                patient_id_list += patient_id
                
                bar.next(len(label))

            bar.finish()
        return {
            'features': np.concatenate(feature_list, 0),
            'logits': np.concatenate(logit_list, 0),
            'probabilities': np.concatenate(prob_list, 0),
            'views': view_list,
            'seq_types': seq_type_list,
            'labels': label_list,
            'patient_ids': patient_id_list
        }

    def save(self):
        """Save model checkpoint"""
        model_path = get_general_save_path('{}-{:0>3}-model'.format(self.save_name, self.epoch), 'ptm', checkpoint=True)
        optimizer_path = get_general_save_path('{}-{:0>3}-optimizer'.format(self.save_name, self.epoch), 'ptm', checkpoint=True)

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load(self, restore_epoch, init=True, find_last_epoch=True):
        """Load model checkpoint"""

        checkpoint_dir = get_cwd(checkpoint=True)

        if find_last_epoch:
            if osp.isdir(checkpoint_dir):
                pattern = re.compile(rf"^{re.escape(self.save_name)}-(\d+)-model\.ptm$")
                epochs = []
                for filename in os.listdir(checkpoint_dir):
                    match = pattern.match(filename)
                    if match:
                        epochs.append(int(match.group(1)))
                if epochs:
                    restore_epoch = max(epochs)
                    log.info(f"Found latest checkpoint epoch {restore_epoch} in {checkpoint_dir}")
                else:
                    log.warning(f"No matching checkpoints found in {checkpoint_dir}; using restore_epoch={restore_epoch}")
            else:
                log.warning(f"No checkpoint directory found in {checkpoint_dir}; using restore_epoch={restore_epoch}")

        self.restore_epoch = restore_epoch
        model_path = get_general_save_path('{}-{:0>3}-model'.format(self.save_name, self.restore_epoch), 'ptm', checkpoint=True)
        optimizer_path = get_general_save_path('{}-{:0>3}-optimizer'.format(self.save_name, self.restore_epoch), 'ptm', checkpoint=True)

        if osp.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = normalize_state_dict_for_load(
                state_dict,
                use_cpu=conf.get("USE_CPU", False),
                model=self.model,
                add_module_prefix=True,
            )
            self.model.load_state_dict(state_dict)
            log.info(f'Model checkpoint loaded from {model_path}')
        else:
            log.warning(f'Model checkpoint not found at {model_path}')
        
        if osp.exists(optimizer_path):
            optimizer_state = torch.load(optimizer_path, map_location='cpu')
            self.optimizer.load_state_dict(optimizer_state)
            log.info(f'Optimizer checkpoint loaded from {optimizer_path}')
        else:
            log.warning(f'Optimizer checkpoint not found at {optimizer_path}')

    def load_pretrained(self, restore_iter):
        """Load pre-trained encoder weights from a different dataset checkpoint.
        Useful for fine-tuning. Only loads encoder weights, optimizer is initialized fresh.
        """
        # Try to find CASIA-B checkpoint as pre-trained model
        pretrained_names = [
            'GaitSet_CASIA-B_73_False_256_0.2_128_full_30',  # CASIA-B checkpoint
        ]
        
        #for pretrained_name in pretrained_names:
        pretrained_path = conf.get('model', {}).get('pretrained_model', {})['PRETRAINED_PATH']
        """osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(pretrained_name, restore_iter))"""
        if osp.exists(pretrained_path):
            log.info(f'Loading pre-trained model from: {pretrained_path}')
            # Only load the encoder weights, not the full model
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            # Debug: print all keys in checkpoint
            log.debug(f'All checkpoint keys: {list(state_dict.keys())}')
            encoder_weights = normalize_state_dict_for_load(
                state_dict,
                use_cpu=conf.get("USE_CPU", False),
                strip_prefixes=["module", "encoder"],
                model=self.model,
                add_module_prefix=False,
            )
            log.debug(f'Filtered encoder_weights: {list(encoder_weights.keys())}')
            if encoder_weights:
                encoder = self.model.module.encoder if isinstance(self.model, nn.DataParallel) else self.model.encoder
                encoder.load_state_dict(
                    encoder_weights,
                    strict=conf.get('model', {}).get('pretrained_model', {}).get('load_state_dict_strict', True),
                )
                log.info('Pre-trained encoder loaded successfully.')
                if not conf.get('model', {}).get('pretrained_model', {}).get('load_state_dict_strict', True):
                    log.warning('Using strict=False for encoder.load_state_dict')
                return
        
        log.warning(f'No pre-trained checkpoint found. Starting from scratch.')

    

