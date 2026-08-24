import os

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch


class DashboardPlotter:

    @staticmethod
    def plot_metrics(metrics, save_path, conf, title="", eval_epoch=0):
        """
        Generate an image using matplotlib in a slide format 16:9 (13.333 x 7.5 in, 
        PowerPoint / Google Slides standard size) to present all the metrics.
        The footer shows a condensed summary of model hyperparameters.
        """

        # -------------------------
        # Metrics
        # -------------------------
        acc = metrics.accuracy * 100
        prec = metrics.precision * 100
        rec = metrics.recall * 100
        spec = metrics.specificity * 100
        f1 = metrics.f1 * 100
        auc = metrics.auc

        tp = metrics.tp
        fp = metrics.fp
        tn = metrics.tn
        fn = metrics.fn

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

        n_epochs = eval_epoch if eval_epoch > 0 else conf['model']['num_epochs']
        pretrained_model = os.path.split(conf['model']['pretrained_model']['PRETRAINED_PATH'])[1]
        optimizer_name = conf['model']['optimizer']['type']
        lr = conf['model']['optimizer']['encoder_lr']
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
                opt_txt += (f" (encoder_lr={lr},wd={weight_decay})") if weight_decay else f" (encoder_lr={lr})"
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
        plt.savefig(save_path, dpi=200, facecolor="white")