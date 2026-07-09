from contextlib import redirect_stdout
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from zoneinfo import ZoneInfo
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import yaml
import pytz
import atexit
from config import load_config, set_conf, conf
from opts import get_opts, get_opts_dic
from metrics_utils import (
    compute_and_save_by_view,
    compute_metrics,
    plot_metrics_by_view,
    select_best_threshold,
    aggregate_by_patient,
    save_metrics,
    print_metrics,
    save_metrics_png,
    roc_curve_output,
    save_result_bundle,
    plot_combined_roc
)


def excepthook(exc_type, exc_value, exc_tb):
    logging.getLogger(__name__).exception(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )

def get_cwd(checkpoint=False, common=False, results=False):
    if 'RUN_ID' not in conf:
        paris_tz = pytz.timezone('Europe/Paris')
        conf['RUN_ID'] = datetime.now(paris_tz).strftime("%Y-%m-%d_%H.%M.%S")
    
    launch_dt = conf['RUN_ID']
    opt = get_opts()
    metadata = conf.get('_metadata', None)

    # --- main work path ---
    path = conf['WORK_PATH']

    # --- debug option ---
    if opt.debug:
        path = os.path.join(path, "debug")
    else:
        path = os.path.join(path, "exp")
    
    # --- subfolder in configuration file ---
    subfolder = conf.get("SUBFOLDER", None)
    if subfolder:
        path = os.path.join(path, subfolder)

    # --- experiment folder ---
    restore_dir = conf.get("RESTORE_DIR", None)
    if restore_dir is not None:
        if os.path.isabs(restore_dir):
            path = restore_dir
        else:
            path = os.path.join(path, restore_dir)
    elif metadata is not None and metadata.has_multiple_configs():
        path = os.path.join(path, metadata.get_parent_folder_name())
    else:
        path = os.path.join(path, "checkpoint_" + conf['model']['model_name'] + "_" + launch_dt)
    
    # --- only for multiple configurations ---
    if metadata is not None and metadata.has_multiple_configs():        
        if results:
            path = os.path.join(path, "results")
        elif common:
            path = os.path.join(path, "common")
        elif metadata.current_config_id is not None:
            path = os.path.join(path, f"config_{metadata.current_config_id}")
        else:
            # During initialization, use a common folder for shared files
            path = os.path.join(path, "common")
    
    # --- checkpoint for model saving ---
    if checkpoint:
        path = os.path.join(path, "checkpoint")

    # --- only during testing option ---
    if opt.test and not opt.train and not common and not checkpoint:
        path = os.path.join(path, "metrics_" + launch_dt)
    
    os.makedirs(path, exist_ok=True)
    return path

def get_general_save_path(name, format, checkpoint=False, common=False, results=False):
    return os.path.join(
        get_cwd(checkpoint=checkpoint, common=common, results=results),
        name + "." + format)

def save_config_and_args():
    config_list = opt.config if isinstance(opt.config, list) else [opt.config]
    data = {"args": {
                "config_files": config_list,
                "training": opt.train,
                "testing": opt.test,
            },
            "data": conf['data'], 
            "model": conf['model'],
            "RESTORE_DIR": conf['RESTORE_DIR'],
            "CUDA_VISIBLE_DEVICES": conf['CUDA_VISIBLE_DEVICES'],
            "USE_CPU": conf['USE_CPU'],
        }
    with open(get_general_save_path("config_and_args", "yaml"), 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

def print_summary(model, b_std_out, b_file):
    if not b_file and not b_std_out:
        return
    # Print model summary with detailed information
    try:
        # Create a sample input tensor matching the expected shape: (batch_size, channels, height, width)
        # For silhouette images: batch_size=4, channels=1 (grayscale), 64x64 spatial dimensions
        if not conf['USE_CPU']:
            sample_input = torch.randn(128, 30, 64, 44).cuda()
            actual_model = model.model.module
        else:
            sample_input = torch.randn(128, 30, 64, 44).to('cpu')
            actual_model = model.model

        from torchinfo import summary
        class Tee:
            def __init__(self, *files):
                self.files = files

            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                return len(obj)

            def flush(self):
                for f in self.files:
                    f.flush()
        
        with open(get_general_save_path("model_summary", "txt"), "w") as f:
            out = []
            if b_std_out:
                out.append(sys.stdout)
            if b_file:
                out.append(f)
            
            with redirect_stdout(Tee(*out)):
                print("\n" + "="*80)
                print(f"MODEL SUMMARY")
                print("="*80)
                summary(
                    actual_model,
                    input_data=sample_input,
                    depth=3,
                    verbose=2,
                    col_names=("input_size", "output_size", "num_params", "params_percent"),
                    row_settings=("var_names",),
                )
    except Exception as e:
        log.warning(f"Warning: Could not generate detailed summary: {e}")
        log.warning(f"Model type: {type(model)}")
        log.warning(f"Actual network type: {type(model.model)}")


def main(conf):
    # Import initialization after parsing options to avoid circular imports
    match conf['model']['model_type']:
        case 1:
            # Use the generic (non-classification) model for CASIA-B
            from model.initialization import initialization # type: ignore
            is_classification_model = False
            # CASIA evaluation helper
            from test_gaitset import print_and_save_results  # type: ignore
        case 2:
            # Use the classification model for D-Gait (and others)
            from model_classification.initialization import initialization # type: ignore
            is_classification_model = True

    log.info('Initializing...')
    model = initialization(conf, train=conf['data']['cache'], test=conf['data']['cache'])[0]

    if opt.train:
        print_summary(model, opt.summary, conf['model']['save_summary'])
        save_config_and_args()

        log.info('Fitting for {} epochs...'.format(model.num_epochs))
        
        train_time = datetime.now()

        try:
            model.fit()
            log.info('Fitting complete. Cost: {}'.format(datetime.now() - train_time))
        except KeyboardInterrupt:
            print("\n")
            log.critical(f"Fitting interrupted at epoch {model.epoch}.")
        finally:
            model.save()
    else:
        log.info("Skipping training or fine-tuning.")

    if opt.test:
        if not is_classification_model:
            log.info('Running evaluation and saving results...')
            output_path = get_general_save_path('casia_eval', 'txt')
            try:
                print_and_save_results(output_path, model, iteration=model.restore_iter, batch_size=conf.get('embeddings_batch_size', 1), cache=conf['data'].get('cache', False), cfg=conf)
            except Exception as e:
                log.error('Evaluation failed: %s', e.with_traceback())
        else:
            if opt.train:
                log.info(f"Loading best model for testing from epoch {model.best_val_f1_epoch}")
                model.load(model.best_val_f1_epoch, init=False)
            
            log.info("Starting evaluation...")
            log.info('Extracting train embeddings...')
            time = datetime.now()
            train_result = model.transform('train', conf['model']['embeddings_batch_size'])
            train_prob = train_result['probabilities'][:,1]
            train_patient = train_result['patient_ids']  # Use actual patient IDs

            log.info('Extracting test embeddings...')
            test_result = model.transform('test', conf['model']['embeddings_batch_size'])
            test_prob = test_result['probabilities'][:,1]
            test_patient = test_result['patient_ids']  # Use actual patient IDs
            log.info(f'Embedding extraction complete. Cost: {datetime.now() - time}')

            train_y = np.array([train_result['labels'][i] for i in range(len(train_result['labels']))], dtype='int32')
            test_y = np.array([test_result['labels'][i] for i in range(len(test_result['labels']))], dtype='int32')

            log.info(f'Train sequences:  {len(train_y)}, patients: {len(set(train_patient))}, D: {int(train_y.sum())}, N: {int((train_y == 0).sum())}')
            log.info(f'Test sequences:  {len(test_y)}, patients: {len(set(test_patient))}, D: {int(test_y.sum())}, N: {int((test_y == 0).sum())}')

            seq_train_metrics = compute_metrics(train_y, train_prob)
            #train_threshold, train_threshold_metrics = select_best_threshold(train_y, train_prob)
            seq_test_metrics = compute_metrics(test_y, test_prob)#, threshold=train_threshold)
            print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics)
            print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics)
            # --- Per-view metrics (sequence-level) for train and test ---
            try:
                train_views = train_result.get('views', [])
                test_views = test_result.get('views', [])

                train_view_metrics, train_view_metrics_path = compute_and_save_by_view(train_views, train_y, train_prob, 'Sequence-level train', get_general_save_path(f"{'Sequence-level train'.replace(' ', '_').lower()}_by_view", "txt"))
                test_view_metrics, test_view_metrics_path = compute_and_save_by_view(test_views, test_y, test_prob, 'Sequence-level test', get_general_save_path(f"{'Sequence-level test'.replace(' ', '_').lower()}_by_view", "txt"))

                log.info(f'Per-view train metrics saved: {train_view_metrics_path}')
                log.info(f'Per-view test metrics saved: {test_view_metrics_path}')
                # --- Plot per-view bar chart for test set ---
                try:
                    out_png = get_general_save_path('per_view_test_metrics', 'png')
                    plot_metrics_by_view(test_view_metrics, out_png, title='Per-view metrics (Test set)')
                    log.info(f'Per-view test metrics plot saved: {out_png}')
                except Exception as e:
                    log.error(f'Error while plotting per-view metrics: {e}')
            except Exception as e:
                log.error(f'Error while computing per-view metrics: {e}')
            #log.info(f'Sequence-level threshold selected on train set: {train_threshold:.3f} (train F1={train_threshold_metrics["f1_D"]:.3f})')
            
            train_patients, train_patient_y, train_patient_prob = aggregate_by_patient(train_patient, train_y, train_prob)
            test_patients, test_patient_y, test_patient_prob = aggregate_by_patient(test_patient, test_y, test_prob)
            patient_train_metrics = compute_metrics(train_patient_y, train_patient_prob)
            #patient_threshold, patient_threshold_metrics = select_best_threshold(train_patient_y, train_patient_prob)
            patient_test_metrics = compute_metrics(test_patient_y, test_patient_prob) #, threshold=patient_threshold)

            print_metrics('=== Patient-level metrics: train ===', patient_train_metrics)
            print_metrics('=== Patient-level metrics: test ===', patient_test_metrics)
            #log.info(f'Patient-level threshold selected on train set: {patient_threshold:.3f} (train F1={patient_threshold_metrics["f1_D"]:.3f})')

            save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics, get_general_save_path("metrics", "txt"))

            patient_roc = roc_curve_output(test_patient_y, test_patient_prob, "_patient", output_file=get_general_save_path("patient_roc_curve", "png"))
            seq_roc = roc_curve_output(test_y, test_prob, "_seq", output_file=get_general_save_path("seq_roc_curve", "png"))

            save_metrics_png(seq_test_metrics, conf, get_general_save_path("seq_test_metrics", "png"), best_val_epoch=model.best_val_f1_epoch, title='=== Sequence-level metrics: test ===')

            # Save bundle for later aggregation
            save_result_bundle(conf['_metadata'].current_config_id,
                               seq_test_metrics, patient_test_metrics,
                               test_y, test_prob, test_patient_y, test_patient_prob,
                               get_general_save_path(f"metrics_config_{conf['_metadata'].current_config_id}", "json", results=True), get_general_save_path(f"arrays_config_{conf['_metadata'].current_config_id}", "npz", results=True))

            # Prepare result dict to return to caller for aggregation
            result = {
                'config_id': conf['_metadata'].current_config_id,
                'label': conf["_metadata"].current_config_id,
                'seq_test_metrics': seq_test_metrics,
                'patient_test_metrics': patient_test_metrics,
                'seq_roc': seq_roc,
                'patient_roc': patient_roc,
            }
            return result

class ParisFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Europe/Paris"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

class MainMetadata:
    """Stores execution metadata for managing multiple config runs."""
    
    def __init__(self, launch_datetime, config_list):
        """Initialize metadata for execution.
        
        Args:
            launch_datetime: ISO format datetime string for this execution session
            config_list: List of config IDs being executed
            work_path: Base work path from config
            subfolder: Optional subfolder from config
        """
        self.launch_datetime = launch_datetime
        self.config_list = config_list
        self.current_config_id = None
        self.parent_folder_name = None
    
    def has_multiple_configs(self):
        """Return True if executing multiple configurations."""
        return len(self.config_list) > 1
    
    def set_current_config_id(self, config_id):
        """Set the current config ID being processed."""
        self.current_config_id = config_id
    
    def get_parent_folder_name(self):
        """Get or create the parent folder name for multiple configs.
        
        Returns:
            Name of parent folder (created only for multiple configs)
        """
        if self.parent_folder_name is None:
            def first_char(s):
                return str(s)[0]
            config_names = "_".join(map(first_char, self.config_list))
            self.parent_folder_name = f"all_exp_{config_names}_{self.launch_datetime}"
        return self.parent_folder_name

if __name__ == '__main__':
    sys.excepthook = excepthook
    # Parse arguments first
    opt = get_opts()
    
    # Ensure config is a list
    if isinstance(opt.config, str):
        opt.config = [opt.config]
    
    # Load environment and data configs (separate YAML files) to avoid reloading
    # everything when switching model configs. Fall back to old behaviour
    # if the new files are not found.
    temp_logger = logging.getLogger(__name__)
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')

    def _load_yaml_file(path):
        def tuple_constructor(loader, node):
            return tuple(loader.construct_sequence(node))
        yaml.SafeLoader.add_constructor('!tuple', tuple_constructor)
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    env_conf = {}
    data_conf = {}
    initial_model_conf = {}

    # ensure opt.config is a list
    if isinstance(opt.config, str):
        opt.config = [opt.config]

    try:
        env_path = os.path.join(CONFIG_DIR, 'env_config.yaml')
        data_path = os.path.join(CONFIG_DIR, 'data_config.yaml')
        if os.path.exists(env_path):
            temp_logger.info(f'Loading environment config from {env_path}')
            env_conf = _load_yaml_file(env_path)
        if os.path.exists(data_path):
            temp_logger.info(f'Loading data config from {data_path}')
            data_conf = _load_yaml_file(data_path)

        # Load initial model config minimally to aid logging setup when available
        if opt.config:
            temp_logger.info(f'Loading initial model configuration config_{opt.config[0]}.yaml')
            initial_model_conf = load_config(opt.config[0])

        # Merge: env <- data <- model
        merged = {}
        merged.update(env_conf)
        merged.update(data_conf)
        merged.update(initial_model_conf)
        set_conf(merged)
    except Exception as e:
        temp_logger.warning(f'Could not load separate env/data config ({e}), falling back to old config loader')
        temp_logger.info(f'Loading configuration from config_{opt.config[0]}.yaml')
        loaded_conf = load_config(opt.config[0])
        set_conf(loaded_conf)  # Set global conf in config_manager
    
    # Create launch_datetime once for all configs
    paris_tz = pytz.timezone('Europe/Paris')
    if 'RUN_ID' not in conf:
        conf['RUN_ID'] = datetime.now(paris_tz).strftime("%Y-%m-%d_%H.%M.%S")

    # Create execution metadata (only creates parent folder if multiple configs)
    metadata = MainMetadata(
        launch_datetime=conf['RUN_ID'],
        config_list=opt.config
    )
    conf['_metadata'] = metadata
    conf['RESTORE_DIR'] = opt.restore_dir
    #Setup devices
    if not conf.get("USE_CPU", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = conf.get("CUDA_VISIBLE_DEVICES", "0,1")
        import torch
    else:
        import torch
        torch.set_default_device('cpu')

    # Setup logging
    root = logging.getLogger()
    fmt = ParisFormatter('[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [%(funcName)s]: %(message)s', datefmt='%d-%m %H:%M:%S')

    # console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    # file (rotates at 5MB, keep 5 backups)
    log_dir = get_cwd()
    launch_datetime = conf['RUN_ID']
    file_path = os.path.join(log_dir, f'run_{launch_datetime}.log')
    file_handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(fmt)

    # replace handlers to avoid duplicates
    root.handlers = []
    root.addHandler(stream_handler)
    root.addHandler(file_handler)
    
    root.setLevel(opt.log_level)
    
    log = logging.getLogger(__name__)
    log.info(f'Configuration loaded successfully from config_{opt.config[0]}.yaml')
    
    @atexit.register
    def goodbye():
        logging.info("Program ended")

    # Execute each configuration sequentially and collect results
    aggregated_results = []
    for i, config_id in enumerate(opt.config, 1):
        log.info('\n')
        log.info(f'{"="*80}')
        log.info(f'Executing configuration {i}/{len(opt.config)}: config_{config_id}.yaml')
        log.info(f'{"="*80}')

        # Set the current config ID for path creation
        metadata.set_current_config_id(config_id)

        # Load configuration for this iteration
        try:
            loaded_conf = load_config(config_id)
        except Exception as e:
            log.error(f'Could not load model config config_{config_id}.yaml: {e}')
            loaded_conf = {}

        # Merge env, data and model-specific configs so we don't reload env/data every run
        merged = {}
        merged.update(env_conf if 'env_conf' in globals() else {})
        merged.update(data_conf if 'data_conf' in globals() else {})
        merged.update(loaded_conf)
        set_conf(merged)
        # Re-apply metadata and RUN_ID to the newly loaded configuration
        conf['_metadata'] = metadata
        conf['RUN_ID'] = metadata.launch_datetime
        conf['RESTORE_DIR'] = opt.restore_dir
        # Execute main with this configuration and collect result
        res = main(conf)
        if res is not None:
            aggregated_results.append(res)

        log.info(f'Configuration config_{config_id}.yaml completed successfully.')

    # If multiple configs were executed and produced results, create combined plots
    if len(aggregated_results) > 1:
        cmp_path = get_general_save_path("comparison_roc_f1", "png", results=True)
        plot_combined_roc(aggregated_results, cmp_path)
        log.info(f'Combined comparison saved: {cmp_path}')

    log.info(f'{"="*80}')
    log.info(f'All {len(opt.config)} configuration(s) executed successfully!')
    log.info(f'{"="*80}')
