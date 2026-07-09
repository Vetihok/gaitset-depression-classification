"""Shared options container to avoid circular imports.

Usage:
    from opts import get_opts, set_opts
    opt = get_opts()  # may be None until set by the main script

The main entrypoint (`classify_depression.py`) should call `set_opts(opt)`
after parsing command-line arguments. Other modules can call `get_opts()`
to access the parsed options without importing `classify_depression`.
"""

import logging

opt = None


def set_opts(o):
    global opt
    opt = o


def get_opts(parse_if_missing=True, defaults=None):
    """Return existing opts or lazily create them.

    - If `opt` is already set, return it.
    - If `parse_fn` is provided, call it to produce an argparse.Namespace,
      store it via `set_opts()` and return it.
    - If `defaults` (dict) is provided, construct a simple Namespace from
      that dict, store and return it.

    This avoids importing the main script while still allowing callers to
    obtain a usable `opt` object.
    """
    global opt
    if opt is not None:
        return opt
    if parse_if_missing:
        parsed = parse_args()
        set_opts(parsed)
        return parsed
    if defaults is not None:
        from types import SimpleNamespace
        opt = SimpleNamespace(**defaults)
        return opt
    return None


def get_opts_dic():
    opt = get_opts()
    if opt is None:
        return {}
    return dict(vars(opt))


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def log_level_string(s):
    if s.upper() not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL'}:
        raise ValueError(f'{s} is not a valid logging level.')
    match s:
        case 'DEBUG':
            return logging.DEBUG
        case 'INFO':
            return logging.INFO
        case 'WARNING':
            return logging.WARNING
        case 'ERROR':
            return logging.ERROR
        case 'CRITICAL':
            return logging.CRITICAL
        case 'FATAL':
            return logging.FATAL


def parse_args():
    import argparse
    import logging as _logging
    parser = argparse.ArgumentParser(description='Fine-tune GaitSet on D-Gait, then classify depression from embeddings')
    parser.add_argument('-c', '--config', dest='config', required=True, type=str, nargs='+',
                        help='Config ID(s) to load (e.g., -c 1 2 3 loads config_1.yaml, config_2.yaml, config_3.yaml). Multiple configs will be executed sequentially. Required.')
    parser.add_argument('--train', dest='train', nargs='?', const=True, default=False, type=boolean_string,
                        help='Set number of iteration, or use --train as a flag (implies TRUE). Default: FALSE')
    parser.add_argument('--test', dest='test', nargs='?', const=True, default=False, type=boolean_string,
                        help='Set TRUE or FALSE, or use --test as a flag (implies TRUE). Default: FALSE')
    parser.add_argument('--get_metrics', dest='get_metrics', nargs='?', const=True, default=False, type=boolean_string,
                        help='Set TRUE or FALSE, or use --get_metrics as a flag (implies TRUE). Default: FALSE')
    parser.add_argument('--debug', dest='debug', nargs='?', const=True, default=False, type=boolean_string,
                        help='No checkpoint directory is created. Everything is saved in a directory called debug, useful for debugging. Set TRUE or FALSE, or use --debug as a flag (implies TRUE). Default: FALSE')
    parser.add_argument('--summary', dest='summary', nargs='?', const=True, default=False, type=boolean_string,
                        help='Show model summary. Set TRUE or FALSE, or use --summary as a flag (implies TRUE). Default: FALSE')
    parser.add_argument('--log_level', default=_logging.INFO, type=log_level_string,
                        help='Logging level can be: DEBUG, INFO, WARNING, ERROR, CRITICAL, FATAL. Default: INFO')
    parser.add_argument('--parse_log', default=None, type=str,
                        help='Log file path to parse. Default: None')
    parser.add_argument('--restore_dir', default=None, type=str,
                        help='Directory to restore. It can be a folder containing multiple experiments or one experiment. Default: None')
    return parser.parse_args()

