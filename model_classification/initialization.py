# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import logging
import os
from copy import deepcopy
import sys

import numpy as np

from classify_depression import get_cwd

from .utils import load_data
from .model import Model
from opts import get_opts

log = logging.getLogger(__name__)
# Determine log level from shared opts if available
"""_opts = get_opts(parse_if_missing=False, defaults={'log_level': logging.INFO})
_level = _opts.log_level if (_opts is not None and hasattr(_opts, 'log_level')) else logging.INFO
log.setLevel(_level)"""
# Modules should not add handlers; the application entrypoint configures them.

def initialize_data(config, train=False, test=False):
    log.info("Initializing data source...")
    train_source, val_source, test_source = load_data(**config['data'])
    if train:
        log.info("Loading training data...")
        train_source.load_all_data()
    if test:
        log.info("Loading test data...")
        test_source.load_all_data()
    log.info("Data initialization complete.")
    return train_source, val_source, test_source


def initialize_model(config, train_source, val_source, test_source):
    log.info("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    opt_cfg = deepcopy(model_config.get('optimizer', {}))
    loss_cfg = deepcopy(model_config.get('loss_config', {}))
    scheduler_cfg = deepcopy(model_config.get('scheduler', {}))
    r_drop_cfg = deepcopy(model_config.get('r_drop', {}))
    dropout_cfg = deepcopy(model_config.get('dropout', {}))
    sampler_cfg = deepcopy(model_config.get('sampler', {}))

    model_param = {
        'model_name': model_config['model_name'],
        'hidden_dim': model_config.get('hidden_dim', 256),

        'num_epochs': model_config.get('num_epochs', 200),
        'restore_epoch': model_config.get('restore_epoch', 0),
        'eval_interval': model_config.get('eval_interval', 1),
        
        'num_workers': model_config.get('num_workers', 0),
        'batch_size': model_config.get('batch_size', (8, 16)),
        #'train_pid_num': data_config['pid_num'],
        'frame_num': model_config.get('frame_num', 30),
        'train_source': train_source,
        'val_source': val_source,
        'test_source': test_source,

        'opt_cfg': opt_cfg,
        'loss_cfg': loss_cfg,
        'sch_cfg': scheduler_cfg,
        'rdrop_cfg': r_drop_cfg,
        'dropout_cfg': dropout_cfg,
        'sampler_cfg': sampler_cfg,
        "early_stop_cfg": deepcopy(model_config.get('early_stop', {})),
        "freeze_cfg": deepcopy(model_config.get('freeze', {}))
    }

    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        #data_config['pid_num'],
        #data_config['pid_shuffle'],
        model_config['hidden_dim'],
        #model_config['margin'],
        batch_size,
        #model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)

    if m.restore_epoch != 0:
        if config.get("RESTORE_DIR", None) is not None:
            log.info(f'Loading model checkpoint at {config.get("RESTORE_DIR", None)}')
            m.load(m.restore_epoch)
        if model_config.get("pretrained_model", {}).get("enabled", False):
            log.warning("Can't load pretrained model if restore_epoch is not equal to 0.")
    else:
        if model_config.get("pretrained_model", {}).get("enabled", False):
            m.load_pretrained(model_config.get("pretrained_model", {})["PRETRAINED_PATH"])

    log.info(f"Model initialization complete. Save name: {model_param['save_name']}")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    log.info("Initialzing...")
    WORK_PATH = get_cwd()
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, val_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, val_source, test_source)