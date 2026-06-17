import logging
import os
import os.path as osp
import pickle
import sys

import numpy as np

from config import conf

from .data_set import DataSet
from opts import get_opts

log = logging.getLogger(__name__)
"""_opts = get_opts(parse_if_missing=False, defaults={'log_level': logging.INFO})
_level = _opts.log_level if (_opts is not None and hasattr(_opts, 'log_level')) else logging.INFO
log.setLevel(_level)"""


def clinical_label(patient_id):
    if patient_id.startswith('D_'):
        return 1
    if patient_id.startswith('N_'):
        return 0
    raise ValueError('Unknown clinical label for patient id: {}'.format(patient_id))

def load_data(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()
    patient_id = list()  # Track patient IDs separately
    
    c = 0
    log.debug(f"{str(sorted(list(os.listdir(dataset_path))))=:.100}")
    for _patient_id in sorted(list(os.listdir(dataset_path))):
        # In CASIA-B, data of subject #5 is incomplete.
        # Thus, we ignore it in training.
        if dataset == 'CASIA-B' and _patient_id == '005':
            continue
        label_path = osp.join(dataset_path, _patient_id)
        if c <= 3:
            c  += 1
            log.debug(f"_patient_id = {type(_patient_id)} = {clinical_label(_patient_id)}")
            log.debug(f"label_path = {label_path}")

        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            for _view in sorted(list(os.listdir(seq_type_path))):
                _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(clinical_label(_patient_id))
                    patient_id.append(_patient_id)  # Store patient ID
                    seq_type.append(_seq_type)
                    view.append(_view)

    log.debug(f"pid_num = {pid_num}, dataset = {dataset}")
    log.debug(f"{str(seq_dir)=:.100}")
    log.debug(f"{str(view)=:.100}")
    log.debug(f"{str(seq_type)=:.100}")
    log.debug(f"{str(label)=:.100}")
    log.debug(f"unique patient_ids = {sorted(list(set(patient_id)))}")
    
    pid_path = osp.join(conf['WORK_PATH'], 'partition')
    os.makedirs(pid_path, exist_ok=True)
    pid_fname = osp.join(pid_path, '{}_{}_{}.pkl'.format(
        dataset, pid_num, pid_shuffle))
    if not osp.exists(pid_fname):
        # Partition by patient ID, not by depression label
        pid_list = sorted(list(set(patient_id)))
        if pid_shuffle:
            np.random.shuffle(pid_list)
        log.debug(f"pid_list length = {len(pid_list)}")
        log.debug(f"pid_list[0:pid_num] = {pid_list[0:pid_num]}")
        log.debug(f"pid_list[pid_num:] = {pid_list[pid_num:]}")
        pid_list = [pid_list[0:pid_num], pid_list[pid_num:]]
        with open(pid_fname, 'wb') as f:
            pickle.dump(pid_list, f)

    log.debug(f"pid_fname = {pid_fname}")
    with open(pid_fname, 'rb') as f:
        pid_list = pickle.load(f)
    train_list = pid_list[0]
    test_list = pid_list[1]
    
    log.debug(f"train_list (patients) = {train_list}")
    log.debug(f"test_list (patients) = {test_list}")
    
    train_source = DataSet(
        [seq_dir[i] for i, p in enumerate(patient_id) if p in train_list],
        [label[i] for i, p in enumerate(patient_id) if p in train_list],
        [seq_type[i] for i, p in enumerate(patient_id) if p in train_list],
        [view[i] for i, p in enumerate(patient_id) if p in train_list],
        cache, resolution,
        [patient_id[i] for i, p in enumerate(patient_id) if p in train_list])
    test_source = DataSet(
        [seq_dir[i] for i, p in enumerate(patient_id) if p in test_list],
        [label[i] for i, p in enumerate(patient_id) if p in test_list],
        [seq_type[i] for i, p in enumerate(patient_id) if p in test_list],
        [view[i] for i, p in enumerate(patient_id) if p in test_list],
        cache, resolution,
        [patient_id[i] for i, p in enumerate(patient_id) if p in test_list])

    log.debug(f"{str(train_source.label)=:.100}")
    log.debug(f"train_source size = {len(train_source)}")
    log.debug(f"test_source size = {len(test_source)}")
    return train_source, test_source
