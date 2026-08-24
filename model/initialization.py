# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import logging
import os
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np

from env_manager import EnvManager

from .utils import load_data
from .model import Model
from opts import get_opts

log = logging.getLogger(__name__)

_DATA_CACHE = {}

def initialize_data(config, train=False, test=False):
    log.info("Initializing data source...")
    
    if config['data']['cache']:
        key = ("train_source", "val_source", "test_source")
        if key in _DATA_CACHE:
            log.debug("Using cached data sources.")
            train_source, val_source, test_source = _DATA_CACHE[key]
        else:
            train_source, val_source, test_source = load_data(**config['data'])
            _DATA_CACHE[key] = train_source, val_source, test_source
            log.debug(f"Saved data sources in cache.")
    else:
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
    classifier_head_cfg = deepcopy(model_config.get('classifier_head', None))

    model_param = {
        'model_name': model_config['model_name'],
        'hidden_dim': model_config.get('hidden_dim', 256),

        'num_epochs': model_config.get('num_epochs', 200),
        'restore_epoch': model_config.get('restore_epoch', 0),
        'eval_interval': model_config.get('eval_interval', 1),
        
        'num_workers': model_config.get('num_workers', 0),
        'batch_size': model_config.get('batch_size', (8, 16)),
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
        'classifier_head_cfg': classifier_head_cfg,
        "early_stop_cfg": deepcopy(model_config.get('early_stop', {})),
        "freeze_cfg": deepcopy(model_config.get('freeze', {}))
    }

    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        model_config['hidden_dim'],
        batch_size,
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    opt = get_opts()
    env = EnvManager.get_instance()

    checkpoint_dir = Path(env.get_dir(checkpoint=True, mkdir=False))
    if (opt.restore_dir or m.restore_epoch != 0):
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            log.info(f'Loading model checkpoint at {opt.restore_dir}')
            m.load(m.restore_epoch, mode=env.restore_mode)
        else:
            log.warning(f'Checkpoint directory not find or empty at {checkpoint_dir} ')
            log.warning(f'No checkpoint loaded.')
    else:
        if model_config.get("pretrained_model", {}).get("enabled", False):
            m.load_pretrained(model_config.get("pretrained_model", {})["PRETRAINED_PATH"])

    log.info(f"Model initialization complete. Save name: {model_param['save_name']}")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    """
    Initialize data sources and model.
    """
    log.info("Initialzing...")
    train_source, val_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, val_source, test_source)