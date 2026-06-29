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


def excepthook(exc_type, exc_value, exc_tb):
    logging.getLogger(__name__).exception(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_tb)
    )

def get_cwd():
    if 'RUN_ID' not in conf:
        paris_tz = pytz.timezone('Europe/Paris')
        conf['RUN_ID'] = datetime.now(paris_tz).strftime("%Y-%m-%d_%H.%M.%S")
    
    launch_dt = conf['RUN_ID']
    opt = get_opts()
    if opt.debug:
        path = os.path.join(conf['WORK_PATH'], "debug")
        os.makedirs(path, exist_ok=True)
        return path
    elif conf.get("RESTORE_DIR", None) is not None:
        if opt.get_metrics:
            path = os.path.join(conf['RESTORE_DIR'], "metrics_" + launch_dt)
            os.makedirs(path, exist_ok=True)
            return path
        return conf['RESTORE_DIR']
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
                      title="Sequence level",
                      n_epochs=None,
                      pretrained_model=None,
                      optimizer_name=None,
                      lr=None,
                      weight_decay=None,
                      class_weights=None,
                      scheduler=None,
                      dropout=None,
                      save_info_txt=True):
    """
    Génère une image au format slide 16:9 (13.333 x 7.5 in, taille standard
    PowerPoint / Google Slides) présentant les métriques de test d'un modèle
    de classification binaire.

    Le pied de page affiche un résumé condensé des paramètres d'entraînement.
    Le détail complet est toujours sauvegardé en parallèle dans un .txt
    (même nom que `output_file`, suffixe "_info.txt") pour ne rien perdre.
    """

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

    def fmt_int(n):
        return f"{n:,}".replace(",", " ")

    # -------------------------
    # Palette
    # -------------------------
    purple = "#7F77DD"
    purple_dark = "#534AB7"
    purple_light = "#EEECFB"
    bg_bar = "#F0F1F3"
    border = "#E5E7EB"

    green_bg, green_txt = "#EAF3DE", "#27500A"
    red_bg, red_txt = "#FAECE7", "#993C1D"

    text_dark = "#111827"
    text_mid = "#4B5563"
    text_light = "#9CA3AF"

    fontsize_title_1 = 18
    fontsize_title_2 = 14
    fontsize_title_3 = 11
    fontsize_text = 12
    fontsize_precesion = 9.5
    fontsize_key_number = 40
    fontsize_key_number_2 = 15
    # -------------------------
    # Figure - format slide 16:9
    # -------------------------
    fig_w, fig_h = 13.333, 7.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # -------------------------
    # Header
    # -------------------------
    ax.add_patch(FancyBboxPatch((0.025, 0.88), 0.011, 0.085,
                                 boxstyle="round,pad=0,rounding_size=0.005",
                                 linewidth=0, facecolor=purple))

    ax.text(0.05, 0.925, title.upper(), fontsize=fontsize_title_1, fontweight="bold", color=text_dark)
    ax.text(0.05, 0.888, f"Test dataset · {fmt_int(n_patients)} sequences",
            fontsize=fontsize_text, color=text_light)

    # ax.text(0.975, 0.925, "AUC", fontsize=10.5, color=text_light, ha="right", fontweight="bold")
    # ax.text(0.975, 0.875, f"{auc:.3f}", fontsize=24, fontweight="bold",
    #         color=purple_dark, ha="right")

    ax.plot([0.035, 0.965], [0.835, 0.835], color=border, linewidth=1)

    # -------------------------
    # Colonne 1 : carte Accuracy (hero metric)
    # -------------------------
    col1_x, col1_w = 0.035, 0.255
    col1_y_bottom = 0.42

    ax.add_patch(FancyBboxPatch((col1_x, col1_y_bottom), col1_w, 0.375,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=0, facecolor=purple_light))
    ax.text(col1_x + 0.025, 0.745, "ACCURACY", fontsize=fontsize_title_2,
            color=purple_dark, fontweight="bold")
    ax.text(col1_x + 0.025, 0.565, f"{acc:.1f}%",
            fontsize=fontsize_key_number, fontweight="bold", color=purple_dark)
    ax.text(col1_x + 0.025, 0.45, f"{fmt_int(tp + tn)} / {fmt_int(n_patients)} correctly classified",
            fontsize=fontsize_text, color=purple_dark, alpha=0.8)

    n_errors = fp + fn
    err_pct = 100 * n_errors / n_patients if n_patients else 0

    ax.add_patch(FancyBboxPatch((col1_x, 0.20), col1_w, 0.175,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=0, facecolor=bg_bar))
    ax.text(col1_x + 0.025, 0.335, "CLASSIFICATION ERRORS", fontsize=fontsize_title_2,
            color=text_light, fontweight="bold")
    ax.text(col1_x + 0.025, 0.255, fmt_int(n_errors),
            fontsize=22, fontweight="bold", color=text_dark)
    ax.text(col1_x + col1_w - 0.02, 0.255, f"({err_pct:.1f}%)",
            fontsize=12, color=text_mid, ha="right")

    # -------------------------
    # Column 2 : Metrics
    # -------------------------
    metrics_list = [
        ("Precision (D)", prec),
        ("Recall (D)", rec),
        ("Specificity (N)", spec),
        ("F1-score (D)", f1),
    ]

    bar_col_x = 0.335
    bar_w = 0.295
    value_x = 0.695

    y_top = 0.735
    dy = 0.145
    bar_h = 0.03

    for i, (name, value) in enumerate(metrics_list):
        y = y_top - i * dy

        ax.text(bar_col_x, y + 0.045, name, fontsize=fontsize_title_2, color=text_mid, fontweight="bold")

        ax.add_patch(FancyBboxPatch((bar_col_x, y), bar_w, bar_h,
                     boxstyle="round,pad=0.002,rounding_size=0.013",
                     linewidth=0, facecolor=bg_bar))
        ax.add_patch(FancyBboxPatch((bar_col_x, y), bar_w * (value / 100), bar_h,
                     boxstyle="round,pad=0.002,rounding_size=0.013",
                     linewidth=0, facecolor=purple))

        ax.text(value_x, y + bar_h / 2, f"{value:.1f}%", fontsize=fontsize_key_number_2,
                fontweight="bold", color=purple_dark, va="center", ha="right")

    # -------------------------
    # Column 3 : Confusion matrix and AUC
    # -------------------------
    cm_col_x = 0.715
    cm_col_w = 0.255
    box_w = (cm_col_w - 0.018) / 2
    box_h = 0.165
    gap = 0.018
    cm_y0 = 0.205

    ax.text(cm_col_x + box_w + gap/2, cm_y0 + box_h*2 + gap*2, "CONFUSION MATRIX", ha="center", fontsize=fontsize_title_3,
            color=text_light, fontweight="bold")

    def cm_box(x, y, label, val, bg, txt):
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                     boxstyle="round,pad=0.006,rounding_size=0.016",
                     linewidth=0, facecolor=bg))
        ax.text(x + box_w / 2, y + box_h - 0.034, label, ha="center", va="center",
                fontsize=10, color=txt, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h / 2 - 0.03, fmt_int(val),
                ha="center", va="center", fontsize=19, fontweight="bold", color=txt)

    top_y = cm_y0 + box_h + gap
    cm_box(cm_col_x, top_y, "True Positive (D)", tp, green_bg, green_txt)
    cm_box(cm_col_x + box_w + gap, top_y, "False Negative (N)", fn, red_bg, red_txt)
    cm_box(cm_col_x, cm_y0, "False Positive (D)", fp, red_bg, red_txt)
    cm_box(cm_col_x + box_w + gap, cm_y0, "True Negative (N)", tn, green_bg, green_txt)

    ax.text(cm_col_x + box_w / 2, cm_y0 - 0.028, "Depressive", ha="center", fontsize=10, color=text_light)
    ax.text(cm_col_x + box_w + gap + box_w / 2, cm_y0 - 0.028, "Normal", ha="center", fontsize=10, color=text_light)
    
    auc_box = FancyBboxPatch(
        (cm_col_x, cm_y0 + box_h*2 + gap*3 + 0.06),
        cm_col_w,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#F3F4F6",
        linewidth=0
    )
    ax.add_patch(auc_box)

    ax.text(cm_col_x + box_w + gap/2, cm_y0 + box_h*2 + gap*3 + 0.06 + 0.08, "AUC",
            fontsize=fontsize_title_3,
            fontweight="bold",
            color=text_light,
            ha="center")

    ax.text(cm_col_x + box_w + gap/2, cm_y0 + box_h*2 + gap*3 + 0.06 + 0.02,
            f"{auc:.3f}",
            fontsize=fontsize_key_number_2,
            fontweight="bold",
            color=purple_dark,
            ha="center")

    # -------------------------
    # Footer : hyperparameter summary
    # -------------------------
    ax.plot([0.035, 0.965], [0.105, 0.105], color=border, linewidth=1)

    parts = []
    if n_epochs is not None:
        parts.append(f"Epochs : {n_epochs}")
    if pretrained_model is not None:
        parts.append(f"Backbone : {pretrained_model}")
    if optimizer_name is not None:
        opt_txt = str(optimizer_name)
        if lr is not None:
            opt_txt += (f" (lr={lr},wd={weight_decay})") if weight_decay else f" (lr={lr})"
        parts.append(f"Optimizer : {opt_txt}")
    if scheduler is not None:
        parts.append(f"Scheduler : {scheduler}")
    if class_weights is not None:
        parts.append(f"Class weights : {class_weights}")
    if dropout is not None:
        if isinstance(dropout, bool):
            dtxt = "activé" if dropout else "désactivé"
        else:
            dtxt = str(dropout)
        parts.append(f"Dropout : {dtxt}")

    footer_txt = "    ·    ".join(parts) if parts else "No training information"

    max_chars_per_line = 100
    if len(footer_txt) > max_chars_per_line and len(parts) > 1:
        mid = len(parts) // 2
        line1 = "    ·    ".join(parts[:mid])
        line2 = "    ·    ".join(parts[mid:])
        ax.text(0.035, 0.075, line1, fontsize=9.5, color=text_light)
        ax.text(0.035, 0.045, line2, fontsize=9.5, color=text_light)
    else:
        ax.text(0.035, 0.06, footer_txt, fontsize=9.5, color=text_light)

    # -------------------------
    # Sauvegarde de l'image
    # -------------------------
    plt.savefig(output_file, dpi=200, facecolor="white")
    plt.show()
    print(f"PNG sauvegardé : {output_file}")

    # -------------------------
    # Sauvegarde du détail complet en txt
    # -------------------------
    if save_info_txt:
        txt_path = os.path.splitext(output_file)[0] + "_info.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== {title} - informations d'entrainement ===\n")
            f.write(f"Epochs              : {n_epochs}\n")
            f.write(f"Modele pre-entraine : {pretrained_model}\n")
            f.write(f"Optimiseur          : {optimizer_name}\n")
            f.write(f"Learning rate       : {lr}\n")
            f.write(f"Weight decay        : {weight_decay}\n")
            f.write(f"Class weights       : {class_weights}\n")
            f.write(f"Scheduler           : {scheduler}\n")
            f.write(f"Dropout             : {dropout}\n")
        print(f"Infos d'entrainement sauvegardees : {txt_path}")

def parse_loss_from_log(log_file_path):
    """Parse log file and extract train/val loss curves."""
    # Initialize lists
    epochs = []
    train_loss = []
    val_loss = []

    # Patterns for train and validation lines
    train_pattern = re.compile(
        r'\[model.py:382\] \[fit\]: Epoch (\d+)/\d+: loss=([\d.]+)'
    )
    val_pattern = re.compile(
        r'\[model.py:435\] \[fit\]: Epoch (\d+)/\d+: val_loss=([\d.]+)'
    )

    with open(log_file_path, 'r') as f:
        for line in f:
            # Check train loss
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                if epoch not in epochs:
                    epochs.append(epoch)
                    train_loss.append(loss)
                    val_loss.append(None)  # Placeholder

            # Check validation loss
            val_match = val_pattern.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                loss = float(val_match.group(2))
                if epoch in epochs:
                    idx = epochs.index(epoch)
                    val_loss[idx] = loss

    return epochs, train_loss, val_loss

def plot_loss_curves(log_file_path, output_path=None):
    """Parse log and plot loss curves."""
    epochs, train_loss, val_loss = parse_loss_from_log(log_file_path)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss plot saved to: {output_path}")
    else:
        plt.show()

    return epochs, train_loss, val_loss

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
    
    plt.savefig(get_general_save_path("roc_curve" + specifier, "png"))

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
                      save_name, xlabel="Epoch", eval_interval=100):
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
                "training": opt.train,
                "testing": opt.test,
                # "use_pretrained": conf.get('model', {}).get('pretrained_model', {})['PRETRAINED_PATH'] if opt.use_pretrained else opt.use_pretrained,
                # "restore": opt.restore,
                # "eval_interval": opt.eval_interval
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

    print_summary(model, opt.summary, conf['model']['save_summary'])
    
    if opt.parse_log:
        log_path = opt.parse_log
        output_plot = get_general_save_path("loss", "png")

        epochs, train_loss, val_loss = plot_loss_curves(log_path, output_plot)
        return

    save_config_and_args()

    if opt.train:
        log.info('Fitting for {} epochs...'.format(model.num_epochs))
        
        train_time = datetime.now()

        try:
            model.fit()
            log.info('Fitting complete. Cost: {}'.format(datetime.now() - train_time))
        except KeyboardInterrupt:
            print("\n")
            log.critical(f"Fitting interrupted at iter {model.epoch + 1}.")
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
            seq_test_metrics = compute_metrics(test_y, test_prob)
            print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics)
            print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics)
            
            train_patients, train_patient_y, train_patient_prob = aggregate_by_patient(train_patient, train_y, train_prob)
            test_patients, test_patient_y, test_patient_prob = aggregate_by_patient(test_patient, test_y, test_prob)
            patient_train_metrics = compute_metrics(train_patient_y, train_patient_prob)
            patient_test_metrics = compute_metrics(test_patient_y, test_patient_prob)

            print_metrics('=== Patient-level metrics: train ===', patient_train_metrics)
            print_metrics('=== Patient-level metrics: test ===', patient_test_metrics)

            save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics)
            roc_curve_output(test_patient_y, test_patient_prob, "_patient")
            roc_curve_output(test_y, test_prob, "_seq")

            save_metrics_png(seq_test_metrics, get_general_save_path("seq_test_metrics", "png"), title='=== Sequence-level metrics: test ===', n_epochs=conf['model']['num_epochs'], 
                             pretrained_model=os.path.split(conf['model']['pretrained_model']['PRETRAINED_PATH'])[1], 
                             optimizer_name=conf['model']['optimizer']['type'], lr=conf['model']['optimizer']['lr'], 
                             weight_decay=conf['model']['optimizer']['weight_decay'], class_weights=conf['model']['loss_config']['cross_entropy']['class_weights'], 
                             scheduler=conf['model']['scheduler']['type'], dropout=str(conf['model']['dropout']['p']*100) + " %" if conf['model']['dropout']['enabled'] else False)
    
    if opt.get_metrics:
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
        seq_test_metrics = compute_metrics(test_y, test_prob)
        print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics)
        print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics)
        
        train_patients, train_patient_y, train_patient_prob = aggregate_by_patient(train_patient, train_y, train_prob)
        test_patients, test_patient_y, test_patient_prob = aggregate_by_patient(test_patient, test_y, test_prob)
        patient_train_metrics = compute_metrics(train_patient_y, train_patient_prob)
        patient_test_metrics = compute_metrics(test_patient_y, test_patient_prob)

        print_metrics('=== Patient-level metrics: train ===', patient_train_metrics)
        print_metrics('=== Patient-level metrics: test ===', patient_test_metrics)

        #save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics)
        roc_curve_output(test_patient_y, test_patient_prob, "_patient")
        roc_curve_output(test_y, test_prob, "_seq")

        save_metrics_png(seq_test_metrics, get_general_save_path("seq_test_metrics", "png"), title='=== Sequence-level metrics: test ===', n_epochs=model.epoch, 
                            pretrained_model=os.path.split(conf['model']['pretrained_model']['PRETRAINED_PATH'])[1], 
                            optimizer_name=conf['model']['optimizer']['type'], lr=conf['model']['optimizer']['lr'], 
                            weight_decay=conf['model']['optimizer']['weight_decay'], class_weights=conf['model']['loss_config']['cross_entropy']['class_weights'], 
                            scheduler=conf['model']['scheduler']['type'], dropout=str(conf['model']['dropout']['p']*100) + " %" if conf['model']['dropout']['enabled'] else False)

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
    
    @atexit.register
    def goodbye():
        logging.info("Program ended")

    main(conf)
