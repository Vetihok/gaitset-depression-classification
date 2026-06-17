import logging
import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import json
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import torch.nn.functional as F

from config import conf
from opts import get_opts

from .network import BinaryClassificationNet
from .utils import ClassificationSampler
from sklearn.metrics import f1_score, recall_score
from classify_depression import plot_metric_curve1

log = logging.getLogger(__name__)
# Determine log level from shared opts if available
"""_opts = get_opts(parse_if_missing=False, defaults={'log_level': logging.INFO})
_level = _opts.log_level if (_opts is not None and hasattr(_opts, 'log_level')) else logging.INFO
log.setLevel(_level)"""
# Modules should not add handlers; the application entrypoint configures them.

class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,  # kept for compatibility but not used
                 margin,  # kept for compatibility but not used
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.frame_num = frame_num
        self.num_workers = num_workers
        
        # Handle batch_size as tuple or int
        if isinstance(batch_size, tuple):
            self.batch_size = batch_size[0]
        else:
            self.batch_size = batch_size
            
        self.model_name = model_name

        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.img_size = img_size

        # Initialize binary classification network
        self.model = BinaryClassificationNet(256).float()
        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        # --- Loss ---
        loss_cfg = conf.get("loss", {})
        raw_weight = loss_cfg.get("weight", None)

        if raw_weight is not None:
            weight = torch.tensor(raw_weight, dtype=torch.float).cuda()
        else:
            weight = None

        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction="mean").cuda()

        # --- R-Drop ---
        rdrop_cfg = conf.get("r_drop", {})
        self.rdrop_enabled = rdrop_cfg.get("enabled", False)
        self.rdrop_alpha = rdrop_cfg.get("alpha", 1.0)
        
        dropout_cfg = conf.get("dropout", {})
    
        if self.rdrop_enabled and not dropout_cfg.get("enabled", False):
            log.warning("R-Drop is activated but dropout is disactivated : R-Drop will have no effect.")

        # --- Optimizer ---
        opt_cfg = conf.get("optimizer", {})
        opt_type = opt_cfg.get("type", "Adam")
        self.lr = opt_cfg.get("lr", self.lr)
        wd = opt_cfg.get("weight_decay", 0.0)

        if opt_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=wd)
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=wd)
        else:
            raise ValueError(f"Optimizer inconnu : {opt_type}")

        # --- Scheduler ---
        sch_cfg = conf.get("scheduler", {})
        sch_type = sch_cfg.get("type", "CosineAnnealingLR")

        if sch_type == "CosineAnnealingLR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                T_max=self.total_iter,
                eta_min=sch_cfg.get("eta_min", 0),
            )
        elif sch_type == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                max_lr=sch_cfg.get("max_lr", self.lr * 10),
                total_steps=self.total_iter,
                pct_start=sch_cfg.get("pct_start", 0.3),
            )
        elif sch_type == "PolynomialLR":
            self.scheduler = optim.lr_scheduler.PolynomialLR(self.optimizer,
                total_iters=self.total_iter,
                power=sch_cfg.get("power", 1.0),
            )
        elif sch_type == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sch_cfg.get("milestones", [10000, 20000, 30000]),
                gamma=sch_cfg.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"Scheduler inconnu : {sch_type}")

        # Metrics tracking
        self.loss_history = []
        self.acc_history = []
        self.f1_history = []
        self.recall_history = []

        self.val_loss_history = []
        self.val_acc_history = []
        self.val_f1_history = []
        self.val_recall_history = []
        self.val_iterations = []  # Track iteration numbers for each validation measurement

        self.history = None
        self.sample_type = 'all'

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
        """if self.restore_iter != 0:
            log.info(f"Loading checkpoint from iteration {self.restore_iter}...")
            self.load(self.restore_iter)
            self.restore_iter = 0"""

        self.model.train()
        self.sample_type = 'random'
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        # Use ClassificationSampler instead of TripletSampler
        sampler = ClassificationSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        best_val_f1 = -1.0
        eval_interval = get_opts().eval_interval
        if eval_interval >= self.total_iter:
            log.warning(f"No evaluation during training : {eval_interval=} >= {self.total_iter=}")
        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame, _ in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            # Convert sequences to tensors
            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            # Get target labels (0=Normal, 1=Depressed)
            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            # --- before (one forward pass) ---
            logits, features = self.model(*seq, batch_frame)
            loss = self.criterion(logits, target_label)

            # --- after (conditionnal R-Drop) ---
            if self.rdrop_enabled:
                logits, features   = self.model(*seq, batch_frame)   # passe 1
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
            # Convert en numpy pour sklearn
            pred_np = pred.cpu().numpy()
            target_np = target_label.cpu().numpy()
            
            # F1 et Recall macro pour gérer binaire/multi-classe
            f1 = f1_score(target_np, pred_np, average='binary', zero_division=0)
            recall = recall_score(target_np, pred_np, average='binary', zero_division=0)
            
            self.loss_history.append(loss.item())
            self.acc_history.append(acc)
            self.f1_history.append(f1)
            self.recall_history.append(recall)
            
            self.scheduler.step()
            # Save checkpoint and print stats
            if self.restore_iter % eval_interval == 0:
                # Evaluate on test set
                self.model.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_f1 = 0.0
                val_recall = 0.0

                try:
                    # with torch.no_grad():
                    outputs = self.transform('test')
                    val_logits = outputs['logits']
                    val_labels = outputs['labels']
                    if len(val_labels) > 0:
                        val_logits_t = torch.from_numpy(val_logits).cuda()
                        # Map labels to indices using train label set
                        target_val = np.array([train_label_set.index(l) for l in val_labels])
                        target_val_t = torch.from_numpy(target_val).long().cuda()
                        val_loss = self.criterion(val_logits_t, target_val_t).item()
                        val_pred = torch.argmax(val_logits_t, dim=1)
                        val_acc = (val_pred == target_val_t).float().mean().item()

                        # Calcul F1 et Recall pour classification binaire/multi-classe
                        val_pred_np = val_pred.cpu().numpy()
                        target_val_np = target_val_t.cpu().numpy()
                        val_f1 = f1_score(target_val_np, val_pred_np, average='binary')
                        val_recall = recall_score(target_val_np, val_pred_np, average='binary')
                    else:
                        log.warning(f'No validation labels found at iteration {self.restore_iter}')
                except Exception as e:
                    log.error(f'Error during validation at iteration {self.restore_iter}: {e}')
                    import traceback
                    log.error(traceback.format_exc())
                    val_loss = 0.0
                    val_acc = 0.0
                    val_f1 = 0.0
                    val_recall = 0.0

                # Record validation metrics
                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)
                self.val_f1_history.append(val_f1)
                self.val_recall_history.append(val_recall)
                self.val_iterations.append(self.restore_iter)

                # Print stats
                avg_loss = np.mean(self.loss_history[-eval_interval:]) if len(self.loss_history) >= eval_interval else np.mean(self.loss_history)
                avg_acc = np.mean(self.acc_history[-eval_interval:]) if len(self.acc_history) >= eval_interval else np.mean(self.acc_history)
                avg_f1 = np.mean(self.f1_history[-eval_interval:]) if len(self.f1_history) >= eval_interval else np.mean(self.f1_history)
                avg_recall = np.mean(self.recall_history[-eval_interval:]) if len(self.recall_history) >= eval_interval else np.mean(self.recall_history)

                log.info(f'iter {self.restore_iter}: loss={avg_loss:.8f}, acc={avg_acc:.8f}, f1={avg_f1:.8f}, recall={avg_recall:.8f}, '
                        f'val_loss={val_loss:.8f}, val_acc={val_acc:.8f}, val_f1={val_f1:.8f}, val_recall={val_recall:.8f}, '
                        f'lr={self.optimizer.param_groups[0]["lr"]:.6f}, {best_val_f1=:.6f}')

                # Mise à jour history dict
                self.history = {
                    'loss': self.loss_history,
                    'acc': self.acc_history,
                    'f1': self.f1_history,
                    'recall': self.recall_history,
                    'val_loss': self.val_loss_history,
                    'val_acc': self.val_acc_history,
                    'val_f1': self.val_f1_history,
                    'val_recall': self.val_recall_history,
                    'val_iterations': self.val_iterations
                }

                # Plot les 4 courbes
                plot_metric_curve1(self.loss_history, self.val_loss_history, self.val_iterations,
                                metric_name="Loss", save_name="loss", eval_interval=eval_interval)
                plot_metric_curve1(self.acc_history, self.val_acc_history, self.val_iterations,
                                metric_name="Accuracy", save_name="accuracy", eval_interval=eval_interval)
                plot_metric_curve1(self.f1_history, self.val_f1_history, self.val_iterations,
                      metric_name="F1-Score", save_name="f1_score", eval_interval=eval_interval)
                plot_metric_curve1(self.recall_history, self.val_recall_history, self.val_iterations,
                      metric_name="Recall", save_name="recall", eval_interval=eval_interval)

                # Save only if validation f1-score improved
                if val_f1 > best_val_f1:
                    self.save()
                    log.warning(f"Model saved at iteration {self.restore_iter}: {val_f1=:.6f}, {best_val_f1=:.6f}")
                    best_val_f1 = val_f1

                # Return to train mode
                self.model.train()
            
            if self.restore_iter == self.total_iter:
                self.save()
                log.warning(f"Model saved at iteration {self.restore_iter}")
                break

        log.info("Training completed!")

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        """Extract features and predictions from the model"""
        self.model.eval()
        source = self.test_source if flag == 'test' else self.train_source
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
            for seq, view, seq_type, label, batch_frame, patient_id in data_loader:
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()
                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()

                logits, features = self.model(*seq, batch_frame)
                #log.debug(logits.shape)
                probs = torch.softmax(logits, dim=1)
                #log.debug(probs.shape)
                
                # features shape: [batch_size, num_bins, hidden_dim]
                # Flatten to [batch_size, num_bins * hidden_dim]
                batch_size, num_bins, hidden_dim = features.shape
                features_flat = features.contiguous().view(batch_size, -1)
                
                feature_list.append(features_flat.data.cpu().numpy())
                logit_list.append(logits.data.cpu().numpy())
                prob_list.append(probs.data.cpu().numpy())
                
                view_list += view
                seq_type_list += seq_type
                label_list += label
                patient_id_list += patient_id

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
        os.makedirs('checkpoint', exist_ok=True)
        model_path = osp.join('checkpoint', '{}-{:0>5}-model.ptm'.format(self.save_name, self.restore_iter))
        optimizer_path = osp.join('checkpoint', '{}-{:0>5}-optimizer.ptm'.format(self.save_name, self.restore_iter))

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Record last checkpoint details (iteration, paths, timestamp)
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            last_file = os.path.join(root_dir, 'last_checkpoint.txt')
            info = {
                'iteration': int(self.restore_iter),
                'model_path': os.path.abspath(model_path),
                'optimizer_path': os.path.abspath(optimizer_path),
                'timestamp': datetime.now().isoformat()
            }
            with open(last_file, 'w') as f:
                json.dump(info, f)
        except Exception:
            pass

    def load(self, restore_iter, resume_checkpoint=False):
        """Load model checkpoint"""
        # If restore_iter is negative, try to read last checkpoint details
        if restore_iter < 0:
            try:
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                last_file = os.path.join(root_dir, 'last_checkpoint.txt')
                if osp.exists(last_file):
                    with open(last_file, 'r') as f:
                        content = f.read().strip()
                        try:
                            info = json.loads(content)
                        except Exception:
                            # fallback: plain integer in file
                            info = {'iteration': int(content)}

                        if 'iteration' in info:
                            self.restore_iter = int(info['iteration'])

                        if 'model_path' in info and os.path.exists(info['model_path']):
                            model_path = info['model_path']
                            optimizer_path = info.get('optimizer_path')
                            if optimizer_path and not os.path.exists(optimizer_path):
                                optimizer_path = None
                            log.info(f"Loading model from saved path: {model_path}")
                            self.model.load_state_dict(torch.load(model_path))
                            if optimizer_path:
                                self.optimizer.load_state_dict(torch.load(optimizer_path))
                            return
                        elif 'iteration' in info:
                            self.restore_iter = restore_iter
                            restore_iter = int(info['iteration'])
                            log.info(f'Found last checkpoint iteration: {restore_iter}')
                        else:
                            log.warning('last_checkpoint.txt does not contain usable data; cannot restore last checkpoint.')
                            return
                else:
                    log.warning('last_checkpoint.txt not found; cannot restore last checkpoint.')
                    return
            except Exception as e:
                log.warning(f'Could not read last_checkpoint.txt: {e}')
                return

        self.restore_iter = restore_iter
        if resume_checkpoint:
            model_path = osp.join(conf['RESUME_DIR'], "checkpoint", '{}-{:0>5}-model.ptm'.format(self.save_name, restore_iter))
            optimizer_path = osp.join(conf['RESUME_DIR'], "checkpoint", '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))
        else:
            model_path = osp.join(conf['RESTORE_DIR'], '{}-{:0>5}-model.ptm'.format(self.save_name, restore_iter))
            optimizer_path = osp.join(conf['RESTORE_DIR'], '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))
        
        if osp.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            log.info(f'Model checkpoint loaded from {model_path}')
        else:
            log.warning(f'Model checkpoint not found at {model_path}')
        
        if osp.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            log.info(f'Optimizer checkpoint loaded from {optimizer_path}')

    def load_pretrained(self, restore_iter):
        """Load pre-trained encoder weights from a different dataset checkpoint.
        Useful for fine-tuning. Only loads encoder weights, optimizer is initialized fresh.
        """
        # Try to find CASIA-B checkpoint as pre-trained model
        pretrained_names = [
            'GaitSet_CASIA-B_73_False_256_0.2_128_full_30',  # CASIA-B checkpoint
        ]
        
        #for pretrained_name in pretrained_names:
        pretrained_path = conf['PRETRAINED_PATH']
        """osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(pretrained_name, restore_iter))"""
        if osp.exists(pretrained_path):
            log.info(f'Loading pre-trained model from: {pretrained_path}')
            # Only load the encoder weights, not the full model
            checkpoint = torch.load(pretrained_path)
            # Debug: print all keys in checkpoint
            log.debug(f'All checkpoint keys: {list(checkpoint.keys())}')
            # Load all checkpoint keys and remove 'module.' prefix
            encoder_weights = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            log.debug(f'Filtered encoder_weights: {list(encoder_weights.keys())}')
            if encoder_weights:
                self.model.module.encoder.load_state_dict(encoder_weights, strict=conf.get('load_state_dict_strict', True))
                log.info('Pre-trained encoder loaded successfully.')
                if not conf.get('load_state_dict_strict', True):
                    log.warning('Using strict=False for encoder.load_state_dict')
                return
        
        log.warning(f'No pre-trained checkpoint found. Starting from scratch.')
