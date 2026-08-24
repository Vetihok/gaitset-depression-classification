from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import atexit

from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import yaml
import pytz

from config import load_config, set_conf, conf, _load_yaml_file
from env_manager import EnvManager
from evaluation.evaluator import ResultBundle
from opts import get_opts, get_opts_dic
from evaluation import Evaluator

def excepthook(exc_type, exc_value, exc_tb):
    logging.getLogger(__name__).exception(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )

def save_config():
    """
    Save config file in config dir if not already saved.
    """
    data = {
            "data": conf['data'], 
            "model": conf['model']
        }
    save_path = env.get_general_save_path("config", "yaml")
    if os.path.exists(save_path):
        log.info("Config file already saved.")
    else:
        with open(save_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

def run_config(conf):
    """
    Run training and testing for the configuration passed in arguments.

    Return a ResultBundle only if testing is activated.

    """
    # Import initialization after parsing options to avoid circular imports
    from model.initialization import initialization # type: ignore

    log.info('Initializing...')
    model = initialization(conf, train=conf['data']['load_all_train_data'], test=conf['data']['load_all_test_data'])[0]

    train_skipped = False
    checkpoint_dir = Path(env.get_dir(checkpoint=True, mkdir=False))

    if opt.train and (not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir())):
        model.print_summary(opt.summary, conf['model']['save_summary'])        

        log.info('Fitting for {} epochs...'.format(model.num_epochs))
        
        train_time = datetime.now()

        try:
            model.fit()
            log.info('Fitting complete. Cost: {}'.format(datetime.now() - train_time))
        except KeyboardInterrupt:
            print("\n")
            log.critical(f"Fitting interrupted at epoch {model.epoch}." + ' Cost: {}'.format(datetime.now() - train_time))
        finally:
            model.save()
    else:
        log.info("Skipping training")
        train_skipped = True

    if opt.test:
        if opt.train and not train_skipped:
            log.info(f"Loading best model for testing from epoch {model.best_val_f1_epoch}")
            model.load(model.best_val_f1_epoch)
        
        log.info("Starting evaluation...")
        time = datetime.now()

        log.info('Extracting test embeddings...')

        _ , _ , test_prob, views, clothing, test_y, patient_ids = \
            model.transform('test', conf['model']['embeddings_batch_size'])

        log.info(f'Embedding extraction complete. Cost: {datetime.now() - time}')

        seq_test_result_bundle = Evaluator.evaluate_model(conf, test_y, test_prob[:, 1], patient_ids, views, clothing, eval_epoch=model.epoch if train_skipped else model.best_val_f1_epoch)

        return seq_test_result_bundle

class ParisFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Europe/Paris"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

if __name__ == '__main__':
    # Parse arguments first
    opt = get_opts()
    
    # --- Setup Environment ---
    if isinstance(opt.config, str):
        opt.config = [opt.config]

    paris_tz = pytz.timezone('Europe/Paris')
    launch_datetime = datetime.now(paris_tz).strftime("%Y-%m-%d_%H.%M.%S")
    env = EnvManager.from_yaml("env_config.yaml", opt.config, launch_datetime)

    # --- Setup logging ---
    root = logging.getLogger()
    fmt = ParisFormatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [%(funcName)s]: %(message)s', datefmt='%d-%m %H:%M:%S')

    # console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    # file (rotates at 5MB, keep 5 backups)
    file_path = os.path.join(env.get_dir(), f'run_{env.launch_datetime}.log')
    file_handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(fmt)

    root.handlers = []
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    root.setLevel(opt.log_level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    sys.excepthook = excepthook
    @atexit.register
    def goodbye():
        logging.info("Program ended")
    
    log = logging.getLogger(__name__)

    
    # --- Setup devices ---
    if not env.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = env.cuda_visible_devices
        import torch
    else:
        import torch
        torch.set_default_device('cpu')

    # --- Setup data configuration ---
    data_conf = {}
    data_path = os.path.join(env.get_configs_dir(), 'data_config.yaml')
    if os.path.exists(data_path):
        log.info(f'Loading data config from {data_path}')
        data_conf = _load_yaml_file(data_path)
    else:
        log.warning(f"Couldn't load data config from: {data_path}")

    completed_configs = env.get_completed_configs() \
        if opt.restore_dir and not env.overwrite_average_results else []
    log.debug(f"Completed configs = {completed_configs}")
    for i, config_id in enumerate(opt.config, 1):
        log.info('\n')
        log.info(f'{"="*80}')
        log.info(f'Executing configuration {i}/{len(opt.config)}: config_{config_id}.yaml ({env.num_runs} runs each)')
        log.info(f'{"="*80}')

        # --- Set the current config ID ---
        env.set_current_config_id(config_id)
        
        # --- Load configuration for this iteration ---
        try:
            loaded_conf = load_config(env, config_id)
        except Exception as e:
            log.error(f'Could not load model config config_{config_id}.yaml: {e}')
            loaded_conf = {}
        merged = {}
        merged.update(data_conf if 'data_conf' in globals() else {})
        merged.update(loaded_conf)
        set_conf(merged)


        # --- Execution ---
        tested_runs = env.get_tested_runs_for_config(config_id) \
            if opt.restore_dir and not env.overwrite_test_results else set()
        
        if len(tested_runs) == env.num_runs:
            log.info(f'All {env.num_runs} runs already completed for config {config_id}; loading for averaging.')
        else:
            save_config()
            # Execute missing runs
            for run_idx in set(range(env.num_runs)) - tested_runs:
                log.info(f'\n=== Run {run_idx + 1}/{env.num_runs} for config {config_id} ===')
                
                # Set current run ID for path creation
                env.set_current_run_id(run_idx)
                
                # Execute main with this configuration and collect result
                res = run_config(conf)
                if res is not None:
                    log.info(f'Run {run_idx + 1}/{env.num_runs} completed successfully.')
                else:
                    log.warning(f'Run {run_idx + 1}/{env.num_runs} returned None.')

        # --- Average results computation ---
        tested_runs = env.get_tested_runs_for_config(config_id)
        if not len(tested_runs) == env.num_runs:
            log.warning(f'Configuration config_{config_id}.yaml: no runs completed.')
        elif config_id in completed_configs:
            log.info(f'Configuration config_{config_id}.yaml: already averaged all runs.')
        elif not env.has_multiple_runs():
            log.info(f'Configuration config_{config_id}.yaml: only one run, skipping average results.')
        else:
            res_bundles = []

            for run_idx in range(env.num_runs):
                env.set_current_run_id(run_idx)

                res = ResultBundle.load(env.get_dir(results=True))

                res_bundles.append(res)

            env.set_current_run_id(None)

            avg_res = Evaluator.evaluate_config(conf, res_bundles)

            if avg_res is not None:
                log.info(f'Configuration config_{config_id}.yaml: averaged {env.num_runs} runs.')
            else:
                log.warning(f'Configuration config_{config_id}.yaml: average returned None.')


    env.set_current_run_id(None)
    env.set_current_config_id(None)

    # --- Configurations comparison ---
    if env.has_multiple_configs():
        res_bundles = []
        
        for config_id in env.config_list:
            env.set_current_config_id(config_id)

            avg_res = ResultBundle.load(env.get_dir(results=True), load_predictions=False)

            res_bundles.append(avg_res)

        env.set_current_config_id(None)
    
        Evaluator.compare(res_bundles)

        log.info(f'Configs compared')

    log.info(f'{"="*80}')
    log.info(f'All {len(opt.config)} configuration(s) executed successfully!')
    log.info(f'{"="*80}')
