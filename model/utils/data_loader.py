import logging
import os
import os.path as osp
import pickle
import sys

import numpy as np
import math

from config import conf

from .data_set import DataSet
from opts import get_opts
from env_manager import EnvManager

log = logging.getLogger(__name__)

def clinical_label(patient_id):
    if patient_id.startswith('D_'):
        return 1
    if patient_id.startswith('N_'):
        return 0
    raise ValueError('Unknown clinical label for patient id: {}'.format(patient_id))


def _split_indices_by_ratio(items, ratios):
    """Slice a list into 3 contiguous chunks per (train, val, test) ratios."""
    n = len(items)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]  # remainder absorbs rounding
    return [train, val, test]


def _stratified_split_patients(pid_list, patient_to_label, ratios, rng):
    """Split patient IDs into train/val/test, preserving each split's class
    proportions (used for split_mode == 'weighted_random')."""
    by_label = {}
    for pid in pid_list:
        by_label.setdefault(patient_to_label[pid], []).append(pid)

    train, val, test = [], [], []
    for pids in by_label.values():
        pids = pids.copy()
        rng.shuffle(pids)
        t, v, te = _split_indices_by_ratio(pids, ratios)
        train.extend(t)
        val.extend(v)
        test.extend(te)
    return [train, val, test]


def load_data(dataset_path, resolution, dataset, partitioning, cache=True, **kwargs):
    """Load dataset samples and partition them into train, validation, and test sets.

        Args:
                dataset_path: Root directory containing one directory per patient or subject.
                resolution: Input resolution passed to each :class:`DataSet`.
                dataset: Dataset name. ``CASIA-B`` excludes subject ``005`` because its data
                        is incomplete.
                partitioning: Mapping containing ``split`` (three ratios summing to 1.0),
                        ``split_mode``, and optionally ``seed``. Supported split modes are:

                                                * ``sequence``: Shuffles and assigns individual sequences. A patient may
                                                    therefore occur in multiple splits.
                                                * ``subject``: Assigns sorted patient IDs contiguously. Patients are kept
                                                    exclusively in one split, but the resulting class balance is not
                                                    explicitly controlled.
                                                * ``random``: Shuffles patient IDs using ``seed`` and assigns them
                                                    contiguously. Patients are kept exclusively in one split, but class
                                                    balance is not explicitly controlled.
                                                * ``weighted_random``: Shuffles patients separately within each clinical
                                                    label and assigns each label group by the requested ratios. Patients
                                                    are kept exclusively in one split and class proportions are
                                                    approximately preserved.

                                                The ratios determine contiguous chunk sizes using rounded counts; any
                                                remainder is assigned to the test split. ``seed`` controls the local
                                                random generator used by ``sequence``, ``random``, and
                                                ``weighted_random``. Existing partition files are reused when present.

                cache: Whether to enable caching in each :class:`DataSet`.
                **kwargs: Additional unused compatibility arguments.

        Returns:
                A ``(train_source, val_source, test_source)`` tuple of :class:`DataSet`
                instances.

        Raises:
                ValueError: If split ratios are invalid or ``split_mode`` is unsupported.

        Notes:
            The returned sources are ``(train_source, val_source, test_source)``. For
            patient-based modes, all samples belonging to a patient are placed in the same
            source; for ``sequence`` mode, samples are selected independently by index.
        """
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    patient_id = list()

    c = 0
    for _patient_id in sorted(os.listdir(dataset_path)):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _patient_id == '005':
            continue
        label_path = osp.join(dataset_path, _patient_id)
        if c <= 3:
            c += 1
        for _seq_type in sorted(os.listdir(label_path)):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(os.listdir(seq_type_path)):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(clinical_label(_patient_id))
                    patient_id.append(_patient_id)
                    seq_type.append(_seq_type)
                    view.append(_view)

    # ---- Partitioning config ----
    cfg = partitioning
    split_ratios = cfg['split']          # (train, val, test)
    split_mode = cfg['split_mode']              # random | weighted_random | subject | sequence
    seed = cfg.get('seed')                      # optional, for reproducibility

    if len(split_ratios) != 3:
        raise ValueError(f"split must have exactly 3 values (train, val, test), got {split_ratios}")
    if not math.isclose(sum(split_ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {split_ratios} (sum={sum(split_ratios)})")

    rng = np.random.RandomState(seed)  # local RNG, doesn't touch global numpy state

    env = EnvManager.get_instance()
    pid_path = osp.join(env.work_path, 'partition')
    os.makedirs(pid_path, exist_ok=True)
    ratio_str = '-'.join(f"{r:.2f}" for r in split_ratios)
    seed_str = f"_seed{seed}" if seed is not None else ""
    pid_fname = osp.join(pid_path, f"{dataset}_{split_mode}_{ratio_str}{seed_str}.pkl")

    if not osp.exists(pid_fname):
        if split_mode == 'sequence':
            # Split at the sequence level — subject grouping is NOT preserved.
            indices = list(range(len(seq_dir)))
            rng.shuffle(indices)
            partition = _split_indices_by_ratio(indices, split_ratios)

        elif split_mode in ('subject', 'random', 'weighted_random'):
            pid_list = sorted(set(patient_id))

            if split_mode == 'subject':
                ordered = pid_list  # deterministic, sorted order

            elif split_mode == 'random':
                ordered = pid_list.copy()
                rng.shuffle(ordered)
                partition = _split_indices_by_ratio(ordered, split_ratios)

            if split_mode == 'subject':
                partition = _split_indices_by_ratio(ordered, split_ratios)

            if split_mode == 'weighted_random':
                patient_to_label = dict(zip(patient_id, label))
                partition = _stratified_split_patients(pid_list, patient_to_label, split_ratios, rng)
        else:
            raise ValueError(f"Unknown split_mode: {split_mode!r}")

        with open(pid_fname, 'wb') as f:
            pickle.dump({'split_mode': split_mode, 'partition': partition}, f)

    with open(pid_fname, 'rb') as f:
        saved = pickle.load(f)
    train_keys, val_keys, test_keys = saved['partition']

    if split_mode == 'sequence':
        def build_source(idxs):
            return DataSet(
                [seq_dir[i] for i in idxs],
                [label[i] for i in idxs],
                [seq_type[i] for i in idxs],
                [view[i] for i in idxs],
                cache, resolution,
                [patient_id[i] for i in idxs])
    else:
        def build_source(pid_subset):
            pid_subset = set(pid_subset)
            mask = [p in pid_subset for p in patient_id]
            return DataSet(
                [s for s, m in zip(seq_dir, mask) if m],
                [l for l, m in zip(label, mask) if m],
                [t for t, m in zip(seq_type, mask) if m],
                [v for v, m in zip(view, mask) if m],
                cache, resolution,
                [p for p, m in zip(patient_id, mask) if m])

    train_source = build_source(train_keys)
    val_source = build_source(val_keys)
    test_source = build_source(test_keys)

    return train_source, val_source, test_source