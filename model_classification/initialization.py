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
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    if train:
        log.info("Loading training data...")
        train_source.load_all_data()
    if test:
        log.info("Loading test data...")
        test_source.load_all_data()
    log.info("Data initialization complete.")
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    log.info("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['total_iter'] = get_opts().train
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    log.info("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    log.info("Initialzing...")
    WORK_PATH = get_cwd()
    os.chdir(WORK_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source)