from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from zoneinfo import ZoneInfo

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


def excepthook(exc_type, exc_value, exc_tb):
    logging.getLogger(__name__).exception(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )

@atexit.register
def goodbye():
    logging.info("Program ended")

def get_cwd():
    if 'RUN_ID' not in conf:
        paris_tz = pytz.timezone('Europe/Paris')
        conf['RUN_ID'] = datetime.now(paris_tz).strftime("%Y-%m-%d_%H.%M.%S")
    
    launch_dt = conf['RUN_ID']
    
    if conf.get("RESUME_DIR", None) is not None:
        return conf['RESUME_DIR']
    else:
        path = os.path.join(conf['WORK_PATH'], "exp", 
                           "checkpoint_" + conf['model']['model_name'] + "_" + launch_dt)
        os.makedirs(path, exist_ok=True)
        return path

def get_general_save_path(name, format):
    return os.path.join(
        get_cwd(),
        name + "." + format)

def roc_auc_score_binary(y_true, y_score):
    y_true = np.asarray(y_true).astype('int32')
    y_score = np.asarray(y_score)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype='float64')
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Average ranks for ties.
    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end

    sum_pos_ranks = ranks[pos].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype('int32')
    y_pred = (np.asarray(y_prob) >= threshold).astype('int32')

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    auc = roc_auc_score_binary(y_true, y_prob)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'sensitivity_recall_D': sensitivity,
        'specificity_N': specificity,
        'f1_D': f1,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }

def aggregate_by_patient(patient_ids, y_true, y_prob):
    grouped = {}
    for patient_id, target, prob in zip(patient_ids, y_true, y_prob):
        grouped.setdefault(patient_id, {'target': target, 'prob': []})
        grouped[patient_id]['prob'].append(prob)

    patient_list = sorted(grouped.keys())
    patient_y = np.array([grouped[p]['target'] for p in patient_list], dtype='int32')
    patient_prob = np.array([np.mean(grouped[p]['prob']) for p in patient_list], dtype='float32')
    return patient_list, patient_y, patient_prob

def save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics):
    with open(get_general_save_path("metrics", "txt"), 'w') as f:
        def print_fn(s):
            f.write(s + '\n')
        print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics, print_fn)
        print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics, print_fn)
        
        print_metrics('=== Patient-level metrics: train ===', patient_train_metrics, print_fn)
        print_metrics('=== Patient-level metrics: test ===', patient_test_metrics, print_fn)

def print_metrics(title, metrics, print_fn=print):
    print_fn(title)
    print_fn('  Accuracy:          {:.3f}'.format(metrics['accuracy'] * 100.0))
    #print_fn('  Balanced accuracy: {:.3f}'.format(metrics['balanced_accuracy'] * 100.0))
    print_fn('  Precision D:       {:.3f}'.format(metrics['precision'] * 100.0))
    print_fn('  Recall D:          {:.3f}'.format(metrics['sensitivity_recall_D'] * 100.0))
    print_fn('  Specificity N:     {:.3f}'.format(metrics['specificity_N'] * 100.0))
    print_fn('  F1 D:              {:.3f}'.format(metrics['f1_D'] * 100.0))
    print_fn('  AUC:               {:.3f}'.format(metrics['auc']))
    print_fn('  Confusion matrix:  TP={}, FP={}, TN={}, FN={}'.format(
        metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']))

def save_metrics_png(metrics,
                     output_file="test_metrics.png",
                     n_patients=None,
                     title="Patient level",
                     n_epochs=None,
                     pretrained_model=None,
                     optimizer_name=None,
                     lr=None,
                     weight_decay=None,
                     class_weights=None,
                     scheduler=None,
                     dropout=None):

    # -------------------------
    # Metrics
    # -------------------------
    acc = metrics['accuracy'] * 100
    prec = metrics['precision'] * 100
    rec = metrics['sensitivity_recall_D'] * 100
    spec = metrics['specificity_N'] * 100
    f1 = metrics['f1_D'] * 100
    auc = metrics['auc']

    tp = metrics['tp']
    fp = metrics['fp']
    tn = metrics['tn']
    fn = metrics['fn']

    if n_patients is None:
        n_patients = tp + fp + tn + fn

    # -------------------------
    # Colors
    # -------------------------
    purple = "#7F77DD"
    purple_dark = "#534AB7"
    bg_bar = "#F3F4F6"
    border = "#D1D5DB"

    green_bg = "#EAF3DE"
    green_txt = "#27500A"
    red_bg = "#FAECE7"
    red_txt = "#993C1D"

    # -------------------------
    # Figure
    # -------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -------------------------
    # Panel + shadow
    # -------------------------
    shadow = FancyBboxPatch(
        (0.045, 0.04),
        0.91,
        0.92,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=0,
        facecolor="black",
        alpha=0.05
    )
    ax.add_patch(shadow)

    panel = FancyBboxPatch(
        (0.04, 0.045),
        0.91,
        0.92,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1,
        edgecolor=border,
        facecolor="white"
    )
    ax.add_patch(panel)

    # -------------------------
    # Header title
    # -------------------------
    ax.text(0.06, 0.94,
            title.upper(),
            fontsize=10,
            color="#9CA3AF",
            fontweight="bold")

    ax.text(0.06, 0.90,
            f"Test · {n_patients} patients",
            fontsize=14,
            fontweight="bold",
            color="#111827")

    # -------------------------
    # Experiment info block (NEW)
    # -------------------------
    info_y = 0.86
    info_x = 0.06
    line_h = 0.03

    def draw_info(label, value, i):
        if value is None:
            value = "N/A"
        ax.text(info_x,
                info_y - i * line_h,
                f"{label}: {value}",
                fontsize=10,
                color="#4B5563")

    draw_info("Epochs", n_epochs, 0)
    draw_info("Pretrained model", pretrained_model, 1)
    draw_info("Optimizer", optimizer_name, 2)

    opt_extra = ""
    if lr is not None:
        opt_extra += f"lr={lr}"
    if weight_decay is not None:
        opt_extra += f", wd={weight_decay}"

    draw_info("Optimizer params", opt_extra if opt_extra else None, 3)

    draw_info("Class weights", class_weights, 4)
    draw_info("Scheduler", scheduler, 5)

    if dropout is not None:
        if isinstance(dropout, bool):
            dropout_txt = "Enabled" if dropout else "Disabled"
        else:
            dropout_txt = f"{dropout}"
    else:
        dropout_txt = None

    draw_info("Dropout", dropout_txt, 6)

    # -------------------------
    # Metrics layout
    # -------------------------
    metrics_list = [
        ("Accuracy", acc),
        ("Recall D", rec),
        ("Specificity N", spec),
        ("Precision D", prec),
        ("F1 D", f1),
    ]

    label_x = 0.06
    bar_x = 0.28
    bar_w = 0.45
    value_x = 0.78

    y_start = 0.58
    dy = 0.07
    bar_h = 0.022

    for i, (name, value) in enumerate(metrics_list):
        y = y_start - i * dy

        ax.text(label_x, y, name,
                fontsize=11,
                color="#4B5563",
                va="center")

        ax.add_patch(FancyBboxPatch(
            (bar_x, y - bar_h / 2),
            bar_w,
            bar_h,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            linewidth=0,
            facecolor=bg_bar
        ))

        ax.add_patch(FancyBboxPatch(
            (bar_x, y - bar_h / 2),
            bar_w * (value / 100),
            bar_h,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            linewidth=0,
            facecolor=purple
        ))

        ax.text(value_x, y,
                f"{value:.1f}%",
                fontsize=11,
                fontweight="bold",
                color=purple_dark,
                va="center")

    # -------------------------
    # AUC
    # -------------------------
    auc_box = FancyBboxPatch(
        (0.06, 0.10),   # ⬅️ déplacé vers le bas
        0.25,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#F3F4F6",
        linewidth=0
    )
    ax.add_patch(auc_box)

    ax.text(0.185, 0.18, "AUC",
            fontsize=10,
            color="#9CA3AF",
            ha="center")

    ax.text(0.185, 0.13,
            f"{auc:.3f}",
            fontsize=18,
            fontweight="bold",
            color=purple_dark,
            ha="center")

    # -------------------------
    # Confusion matrix
    # -------------------------
    cm_x = 0.42
    cm_y = 0.06
    box_w = 0.18
    box_h = 0.09
    gap = 0.02

    def cm_box(x, y, label, val, bg, txt):
        ax.add_patch(FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0,
            facecolor=bg
        ))

        ax.text(x + box_w/2, y + 0.06,
                label, ha="center", va="center",
                fontsize=10, color=txt)

        ax.text(x + box_w/2, y + 0.03,
                str(val), ha="center", va="center",
                fontsize=15, fontweight="bold",
                color=txt)

    cm_box(cm_x, cm_y + box_h + gap, "TP", tp, green_bg, green_txt)
    cm_box(cm_x + box_w + gap, cm_y + box_h + gap, "FN", fn, red_bg, red_txt)
    cm_box(cm_x, cm_y, "FP", fp, red_bg, red_txt)
    cm_box(cm_x + box_w + gap, cm_y, "TN", tn, green_bg, green_txt)

    # -------------------------
    # Save
    # -------------------------
    plt.savefig(output_file,
                dpi=300,
                bbox_inches="tight",
                facecolor="white")

    plt.close(fig)

    print(f"PNG saved to: {output_file}")

def roc_curve_output(target, predictions, specifier=""):
    fpr, tpr, thresholds = roc_curve(target, predictions)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(target, predictions):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.savefig(get_general_save_path("roc_curve", "png"))

def plot_metric_curve1(train_values, val_values, val_iterations, metric_name,
                      save_name, xlabel="Iteration", eval_interval=100):
    
    """Trace and save a train/val curve for any metrics.

    Args:
        train_values: liste des valeurs train à chaque itération
        val_values: liste des valeurs val aux eval points
        val_iterations: liste des itérations où val a été calculée
        metric_name: str, nom de la métrique pour titre/labels, ex: "Loss"
        save_name: str, nom du fichier sans extension, ex: "loss", "accuracy"
        xlabel: str, label axe X
        eval_interval: int, fallback si val_iterations vide"""
    
    if len(train_values) == 0:
        raise ValueError(f'No training data to plot for {metric_name}')

    plt.figure(figsize=(10, 6))

    # Training curve à chaque itération
    plt.plot(range(1, len(train_values) + 1), train_values,
             label=f'Train {metric_name}', linewidth=1.5)

    # Validation curve aux points d'eval
    if len(val_values) > 0:
        if len(val_iterations) > 0 and len(val_iterations) == len(val_values):
            # On utilise les vraies itérations d'eval
            plt.plot(val_iterations, val_values, 'o-',
                    label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)
        else:
            # Fallback: on suppose eval_interval constant
            val_iter_positions = [eval_interval * (i + 1) for i in range(len(val_values))]
            plt.plot(val_iter_positions, val_values, 'o-',
                    label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)

    plt.title(f'Model {metric_name} During Training', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plot_path = get_general_save_path(save_name, "png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path

def plot_metric_curve2(train_values, val_values, val_iterations, metric_name,
                      save_name, xlabel="Iteration", eval_interval=100):
    if len(train_values) == 0:
        raise ValueError(f'No training data to plot for {metric_name}')

    plt.figure(figsize=(10, 6))

    # La courbe train est toujours aux points d'eval (1 valeur par eval_interval)
    # On génère les itérations correspondantes
    train_iterations = [eval_interval * (i + 1) for i in range(len(train_values))]
    plt.plot(train_iterations, train_values,
             label=f'Train {metric_name}', linewidth=1.5)

    if len(val_values) > 0:
        if len(val_iterations) == len(val_values):
            x_val = val_iterations
        else:
            x_val = [eval_interval * (i + 1) for i in range(len(val_values))]
        plt.plot(x_val, val_values, 'o-',
                 label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)

    plt.title(f'Model {metric_name} During Training', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plot_path = get_general_save_path(save_name, "png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path

def save_config_and_args():
    data = {"args": {
                "config_file": opt.config,
                "training_option": opt.train,
                "testing_option": opt.test,
                "use_pretrained": conf['PRETRAINED_PATH'] if opt.use_pretrained else opt.use_pretrained,
                "restore_iter": opt.restore_iter,
                "eval_interval": opt.eval_interval
            },
            "data": conf['data'], 
            "model": conf['model'],
            "optimizer": conf.get("optimizer", {}),
            "scheduler": conf.get("scheduler", {}),
            "loss": conf.get("loss", {}),
            "r_drop": conf.get("r_drop", {}),
            "dropout": conf.get("dropout", {}),
            "other": {
                "model_type": conf['model_type'],
                "embeddings_batch_size": conf['embeddings_batch_size'],
                "cache": conf['cache'],
                "PRETRAINED_PATH": conf['PRETRAINED_PATH'],
                "RESUME_DIR": conf['RESUME_DIR'],
                "CUDA_VISIBLE_DEVICES": conf['CUDA_VISIBLE_DEVICES'],
                "USE_CPU": conf['USE_CPU']
            }}
    with open(get_general_save_path("config_and_args", "yaml"), 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)


def main(conf):
    # Import initialization after parsing options to avoid circular imports
    dataset_name = conf.get('data', {}).get('dataset', '')
    match conf['model_type']:
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
        case 3:
            from model_3.initialization import initialization # type: ignore
            is_classification_model = True
        case 4:
            from model_4.initialization import initialization # type: ignore
            is_classification_model = False

    log.info('Initializing...')
    model = initialization(conf, train=conf['cache'], test=conf['cache'])[0]

    if opt.summary:
        # Print model summary with detailed information
        print("\n" + "="*80)
        print(f"MODEL SUMMARY - Model {conf['model_type']}")
        print("="*80)
        try:
            # Create a sample input tensor matching the expected shape: (batch_size, channels, height, width)
            # For silhouette images: batch_size=4, channels=1 (grayscale), 64x64 spatial dimensions
            sample_input = torch.randn(128, 30, 64, 44).cuda()
            
            from torchinfo import summary
            # Unwrap DataParallel to get the actual model for summary
            actual_model = model.model.module if hasattr(model.model, 'module') else model.encoder.module
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
    
    
    if opt.restore_iter != 0:
        if conf.get("RESUME_DIR", None) is not None:
            log.info('Loading last model checkpoint')
            model.load(opt.restore_iter, resume_checkpoint=True)
        elif opt.restore_iter == -1:
            log.info('Loading last model checkpoint')
            model.load(-1)
        else:
            log.info('Loading model of iteration {}...'.format(opt.restore_iter))
            model.load(opt.restore_iter)

    if opt.use_pretrained:
        #log.info(f'Loading pre-trained model from {conf["PRETRAINED_PATH"]}...')
        model.load_pretrained(opt.use_pretrained)

    save_config_and_args()

    if opt.train > 0:
        log.info('Fitting for {} iterations...'.format(opt.train))
        model.total_iter = opt.train
        
        train_time = datetime.now()

        history = None
        try:
            model.fit()
            log.info('Fitting complete. Cost: {}'.format(datetime.now() - train_time))
        except KeyboardInterrupt:
            print("\n")
            log.critical(f"Fitting interrupted at iter {model.restore_iter}.")
        finally:
            model.save()
            # Only classification models expose history
            if is_classification_model and hasattr(model, 'history'):
                history = model.history
                # Save training/validation loss plot if history is available
                try:
                    if history is not None:
                        train_loss = history.get('loss', [])
                        val_loss = history.get('val_loss', [])
                        val_iterations = history.get('val_iterations', [])
                        
                        if len(train_loss) == 0:
                            raise ValueError('No training loss data to plot')
                        
                        plot_path = plot_metric_curve1(train_loss, val_loss, val_iterations, "Loss", "loss")
                        """plt.figure(figsize=(10, 6))
                        
                        # Plot training loss at every iteration
                        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', linewidth=1.5)
                        
                        # Plot validation loss at specific iterations where it was evaluated
                        if len(val_loss) > 0:
                            if len(val_iterations) > 0:
                                # Use recorded iteration numbers for validation points
                                plt.plot(val_iterations, val_loss, 'o-', label='Validation Loss', 
                                        linewidth=1.5, markersize=6, alpha=0.7)
                            else:
                                # Fallback: assume validation runs every eval_interval iterations
                                eval_interval = 100  # Default interval
                                val_iter_positions = [eval_interval * (i + 1) for i in range(len(val_loss))]
                                plt.plot(val_iter_positions, val_loss, 'o-', label='Validation Loss', 
                                        linewidth=1.5, markersize=6, alpha=0.7)
                        
                        plt.title('Model Loss During Training', fontsize=14)
                        plt.ylabel('Loss', fontsize=12)
                        plt.xlabel('Iteration', fontsize=12)
                        plt.legend(fontsize=11)
                        plt.grid(True, alpha=0.3)
                        
                        plot_path = get_general_save_path("loss", "png")
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()"""
                        log.info(f'Saved loss plot to: {plot_path}')
                        log.info(f'Training iterations: {len(train_loss)}, Validation measurements: {len(val_loss)}')
                except Exception as e:
                    log.warning(f'Could not save loss plot: {e}')
    else:
        log.info("Skipping training or fine-tuning.")

    if opt.test:
        if not is_classification_model:
            log.info('Running evaluation and saving results...')
            output_path = get_general_save_path('casia_eval', 'txt')
            try:
                print_and_save_results(output_path, model, iteration=model.restore_iter, batch_size=conf.get('embeddings_batch_size', 1), cache=conf.get('cache', False), cfg=conf)
            except Exception as e:
                log.error('Evaluation failed: %s', e.with_traceback())
        else:
            log.info("Starting evaluation...")
            log.info('Extracting train embeddings...')
            time = datetime.now()
            train_result = model.transform('train', conf['embeddings_batch_size'])
            train_prob = train_result['probabilities'][:,1]
            train_patient = train_result['patient_ids']  # Use actual patient IDs

            log.info('Extracting test embeddings...')
            test_result = model.transform('test', conf['embeddings_batch_size'])
            test_prob = test_result['probabilities'][:,1]
            test_patient = test_result['patient_ids']  # Use actual patient IDs
            log.info(f'Embedding extraction complete. Cost: {datetime.now() - time}')

            train_y = np.array([train_result['labels'][i] for i in range(len(train_result['labels']))], dtype='int32')
            test_y = np.array([test_result['labels'][i] for i in range(len(test_result['labels']))], dtype='int32')

            log.info(f'Train sequences:  {len(train_y)}, patients: {len(set(train_patient))}, D: {int(train_y.sum())}, N: {int((train_y == 0).sum())}')
            log.info(f'Test sequences:  {len(test_y)}, patients: {len(set(test_patient))}, D: {int(test_y.sum())}, N: {int((test_y == 0).sum())}')

            seq_train_metrics = compute_metrics(train_y, train_prob)
            seq_test_metrics = compute_metrics(test_y, test_prob)
            print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics)
            print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics)
            
            train_patients, train_patient_y, train_patient_prob = aggregate_by_patient(train_patient, train_y, train_prob)
            test_patients, test_patient_y, test_patient_prob = aggregate_by_patient(test_patient, test_y, test_prob)
            patient_train_metrics = compute_metrics(train_patient_y, train_patient_prob)
            patient_test_metrics = compute_metrics(test_patient_y, test_patient_prob)

            print_metrics('=== Patient-level metrics: train ===', patient_train_metrics)
            print_metrics('=== Patient-level metrics: test ===', patient_test_metrics)
            save_metrics_png(patient_test_metrics, get_general_save_path("patient_test_metrics", "png"), title='=== Patient-level metrics: test ===', n_epochs=opt.train, 
                             pretrained_model=os.path.split(conf['PRETRAINED_PATH'])[1], 
                             optimizer_name=conf['optimizer']['type'], lr=conf['optimizer']['lr'], 
                             weight_decay=conf['optimizer']['weight_decay'], class_weights=conf['loss']['weight'], 
                             scheduler=conf['scheduler']['type'], dropout=str(conf['dropout']['p']*100) + " %" if conf['dropout']['enabled'] else False)

            save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics)
            roc_curve_output(test_patient_y, test_patient_prob)
    

class ParisFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, ZoneInfo("Europe/Paris"))
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

if __name__ == '__main__':
    sys.excepthook = excepthook
    # Parse arguments first
    opt = get_opts()
    # Load configuration based on command-line argument
    temp_logger = logging.getLogger(__name__)
    temp_logger.info(f'Loading configuration from config_{opt.config}.yaml')
    loaded_conf = load_config(opt.config)
    set_conf(loaded_conf)  # Set global conf in config_manager
    
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
    log.info(f'Configuration loaded successfully from config_{opt.config}.yaml')

    main(conf)
