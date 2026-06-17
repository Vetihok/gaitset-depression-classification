from datetime import datetime
import logging
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf

"""logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(funcName)s]: %(message)s'
)"""
log = logging.getLogger(__name__)


def evaluate_checkpoint(m, iteration=80000, batch_size=1, cache=False):
    """Transform test set, and run evaluation.

    Returns (acc, duration, transformed_data)
    """
    
    log.info('Transforming...')
    start = datetime.now()
    test = m.transform('test', batch_size)
    duration = datetime.now() - start
    log.info('Evaluating...')
    acc = evaluation(test, conf['data'])
    log.info('Evaluation complete. Cost: %s', duration)
    return acc, duration, test


def evaluate_from_config(model, iteration=80000, batch_size=1, cache=False, cfg=None):
    """Initialize model from config and evaluate a given checkpoint.

    Returns (acc, duration, transformed_data)
    """
    cfg = cfg if cfg is not None else conf
    m = model
    return evaluate_checkpoint(m, iteration=iteration, batch_size=batch_size, cache=cache)


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def de_diag(acc, each_angle=False):
    view_num = acc.shape[0]
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / (view_num - 1)
    if not each_angle:
        result = np.mean(result)
    return result


def print_and_save_results(output_path, model, iteration=80000, batch_size=1, cache=False, cfg=None):
    """Evaluate using `cfg` (or default config), print results, and save them to `output_path`.

    The saved file is a human-readable text summary of the evaluation.
    """
    acc, duration, _ = evaluate_from_config(model, iteration=iteration, batch_size=batch_size, cache=cache, cfg=cfg)

    num_probes = acc.shape[0]
    probe_names = ['NM', 'BG', 'CL', 'BG-CL']  # Add more as needed

    lines = []
    for i in range(1):
        lines.append('===Rank-%d (Include identical-view cases)===' % (i + 1))
        results = ', '.join([f'{name}: {np.mean(acc[p, :, :, i]):.3f}'
                             for p, name in enumerate(probe_names[:num_probes])])
        lines.append(results)

    for i in range(1):
        lines.append('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        results = ', '.join([f'{name}: {de_diag(acc[p, :, :, i]):.3f}'
                             for p, name in enumerate(probe_names[:num_probes])])
        lines.append(results)

    for i in range(1):
        lines.append('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        for p, name in enumerate(probe_names[:num_probes]):
            lines.append(f'{name}: {de_diag(acc[p, :, :, i], each_angle=True)}')

    lines.append(f'Evaluation duration: {duration}')

    # Print to stdout
    for l in lines:
        print(l)

    # Save to file
    try:
        with open(output_path, 'w') as f:
            for l in lines:
                f.write(l + '\n')
        log.info('Saved evaluation summary to %s', output_path)
    except Exception as e:
        log.error('Failed to save evaluation summary to %s: %s', output_path, e)
