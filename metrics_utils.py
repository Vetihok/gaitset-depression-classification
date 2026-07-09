import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score


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


def select_best_threshold(y_true, y_prob, candidate_thresholds=None):
    y_true = np.asarray(y_true).astype('int32')
    y_prob = np.asarray(y_prob).astype('float32')

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(threshold))
        if metrics['f1_D'] > best_f1 + 1e-12:
            best_f1 = metrics['f1_D']
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    return best_threshold, best_metrics


def aggregate_by_patient(patient_ids, y_true, y_prob):
    grouped = {}
    for patient_id, target, prob in zip(patient_ids, y_true, y_prob):
        grouped.setdefault(patient_id, {'target': target, 'prob': []})
        grouped[patient_id]['prob'].append(prob)

    patient_list = sorted(grouped.keys())
    patient_y = np.array([grouped[p]['target'] for p in patient_list], dtype='int32')
    patient_prob = np.array([np.mean(grouped[p]['prob']) for p in patient_list], dtype='float32')
    return patient_list, patient_y, patient_prob


def print_metrics(title, metrics, print_fn=print):
    print_fn(title)
    print_fn('  Accuracy:          {:.3f}'.format(metrics['accuracy'] * 100.0))
    print_fn('  Precision D:       {:.3f}'.format(metrics['precision'] * 100.0))
    print_fn('  Recall D:          {:.3f}'.format(metrics['sensitivity_recall_D'] * 100.0))
    print_fn('  Specificity N:     {:.3f}'.format(metrics['specificity_N'] * 100.0))
    print_fn('  F1 D:              {:.3f}'.format(metrics['f1_D'] * 100.0))
    print_fn('  AUC:               {:.3f}'.format(metrics['auc']))
    print_fn('  Confusion matrix:  TP={}, FP={}, TN={}, FN={}'.format(
        metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']))


def save_metrics(seq_train_metrics, seq_test_metrics, patient_train_metrics, patient_test_metrics, out_path):
    with open(out_path, 'w') as f:
        def print_fn(s):
            f.write(s + '\n')
        print_metrics('=== Sequence-level metrics: train ===', seq_train_metrics, print_fn)
        print_metrics('=== Sequence-level metrics: test ===', seq_test_metrics, print_fn)
        print_metrics('=== Patient-level metrics: train ===', patient_train_metrics, print_fn)
        print_metrics('=== Patient-level metrics: test ===', patient_test_metrics, print_fn)


def save_metrics_png(metrics,
                      conf,
                      output_file="test_metrics.png",
                      title="",
                      save_info_txt=False,
                      best_val_epoch=0):
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

    n_epochs = best_val_epoch if best_val_epoch > 0 else conf['model']['num_epochs']
    pretrained_model = os.path.split(conf['model']['pretrained_model']['PRETRAINED_PATH'])[1]
    optimizer_name = conf['model']['optimizer']['type']
    lr = conf['model']['optimizer']['lr']
    weight_decay = conf['model']['optimizer']['weight_decay']
    class_weights = conf['model']['loss_config']['cross_entropy']['class_weights']
    scheduler = conf['model']['scheduler']['type']
    dropout = " Encoder = " + str(conf['model']['dropout']['encoder_p']*100) + " % ; Head = " + str(conf['model']['dropout']['head_p']*100) + " %" if conf['model']['dropout']['enabled'] else False

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
    #log.info(f"PNG saved : {output_file}")

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
        #log.info(f"Metrics saved in txt file : {txt_path}")


def roc_curve_output(target, predictions, specifier="", output_file=None):
    fpr, tpr, thresholds = roc_curve(target, predictions)
    auc = roc_auc_score(target, predictions)

    if output_file:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve ' + specifier)
        plt.legend(loc="lower right")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()

    return fpr.tolist(), tpr.tolist(), auc


def save_result_bundle(config_id, seq_test_metrics, patient_test_metrics,
                       test_y, test_prob, test_patient_y, test_patient_prob,
                       metrics_path, arrays_path):
    os.makedirs(os.path.split(metrics_path)[0], exist_ok=True)
    os.makedirs(os.path.split(arrays_path)[0], exist_ok=True)
    bundle = {
        'config_id': config_id,
        'seq_test_metrics': seq_test_metrics,
        'patient_test_metrics': patient_test_metrics,
    }
    # Save metrics json
    with open(metrics_path, 'w') as f:
        json.dump(bundle, f, indent=2)

    # Save arrays as npz
    np.savez_compressed(arrays_path,
                        test_y=np.asarray(test_y),
                        test_prob=np.asarray(test_prob),
                        test_patient_y=np.asarray(test_patient_y),
                        test_patient_prob=np.asarray(test_patient_prob))

    return metrics_path, arrays_path


def plot_combined_roc(results_list, save_path):
    """Plot multiple ROC curves on the same figure and a bar chart of test F1s.

    results_list: list of dict with keys: config_id, label, seq_test_metrics, patient_test_metrics,
                  seq_roc (fpr,tpr,auc) and patient_roc (fpr,tpr,auc)
    """
    plt.figure(figsize=(10, 8))

    # Subplot 1: ROC curves (patient-level)
    ax1 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
    for r in results_list:
        label = r.get('label', r['config_id'])
        fpr, tpr, auc = r['patient_roc']
        ax1.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Patient-level ROC (comparison)')
    ax1.legend(loc='lower right')

    # Subplot 2: bar chart of test F1 (patient-level)
    ax2 = plt.subplot2grid((3, 2), (1, 1), colspan=1)
    labels = [r.get('label', r['config_id']) for r in results_list]
    f1s = [r['seq_test_metrics']['f1_D'] for r in results_list]
    ax2.barh(range(len(labels)), [v * 100 for v in f1s], color='C0')
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('F1-score (sequence-level) [%]')
    ax2.set_title('Sequence-level Test F1')

    # Subplot 3: Sequence-level ROC curves
    ax3 = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=2)
    for r in results_list:
        label = r.get('label', r['config_id'])
        fpr, tpr, auc = r['seq_roc']
        ax3.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Sequence-level ROC (comparison)')
    ax3.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return save_path


def plot_metric_curve1(plot_path, train_values, val_values, val_iterations, metric_name,
                    xlabel="Iteration", eval_interval=100):
    
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

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def plot_metric_curve2(plot_path, train_values, val_values, val_iterations, metric_name,
                      xlabel="Epoch", eval_interval=100):
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
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def plot_metrics_by_view(view_metrics, out_png, title="Per-view metrics"):
    views = sorted(view_metrics.keys())
    f1s = [view_metrics[v]['f1_D'] * 100.0 for v in views]
    accs = [view_metrics[v]['accuracy'] * 100.0 for v in views]
    recs = [view_metrics[v]['sensitivity_recall_D'] * 100.0 for v in views]

    x = np.arange(len(views))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(views) * 0.8), 4.5))
    ax.bar(x - width, f1s, width, label='F1-score', color='#7F77DD')
    ax.bar(x, accs, width, label='Accuracy', color='#53A653')
    ax.bar(x + width, recs, width, label='Recall', color='#D9534F')

    ax.set_xticks(x)
    ax.set_xticklabels(views, rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, facecolor='white')
    plt.close(fig)


def compute_and_save_by_view(views, y, prob, title_prefix, out_path):
    views_list = list(views)
    unique_views = sorted(set(views_list))
    view_metrics = {}
    with open(out_path, 'w', encoding='utf-8') as f:
        def _pf(s):
            f.write(s + '\n')
        for v in unique_views:
            idxs = [i for i, vv in enumerate(views_list) if vv == v]
            if len(idxs) == 0:
                continue
            y_v = np.array([y[i] for i in idxs], dtype='int32')
            p_v = np.array([prob[i] for i in idxs], dtype='float32')
            metrics_v = compute_metrics(y_v, p_v)
            view_metrics[v] = metrics_v
            print_metrics(f"=== {title_prefix} - view: {v} (n={len(idxs)}) ===", metrics_v, print_fn=_pf)
    return view_metrics, out_path