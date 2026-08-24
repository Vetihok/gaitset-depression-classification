from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

log = logging.getLogger(__name__)

# ============================================================
# ROC CURVE
# ============================================================

@dataclass
class ROCCurve:
    """
    ROC curve and associated AUC.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    std_tpr: np.ndarray | None = None
    std_auc: float | None = None
    # --------------------------------------------------------

    @classmethod
    def from_pred(cls, y_true, y_prob):

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        return cls(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc=auc,
        )

    @classmethod
    def average(cls, curves, n_points=1000):
        """
        Compute the mean ROC curve from several ROC curves.

        Parameters
        ----------
        curves : list[ROCCurve]

        Returns
        -------
        ROCCurve
        """

        if len(curves) == 0:
            raise ValueError("Cannot average an empty list of ROC curves.")

        common_fpr = np.linspace(0.0, 1.0, n_points)

        interpolated_tpr = []
        aucs = []

        for curve in curves:

            tpr = np.interp(
                common_fpr,
                curve.fpr,
                curve.tpr,
            )

            tpr[0] = 0.0
            tpr[-1] = 1.0

            interpolated_tpr.append(tpr)
            aucs.append(curve.auc)

        interpolated_tpr = np.asarray(interpolated_tpr)

        mean_tpr = interpolated_tpr.mean(axis=0)
        std_tpr = interpolated_tpr.std(axis=0)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        return cls(
            fpr=common_fpr,
            tpr=mean_tpr,
            thresholds=None,
            auc=mean_auc,
            std_auc=std_auc,
            std_tpr=std_tpr
        )

    # --------------------------------------------------------

    def to_dict(self):

        return {
            "fpr": self.fpr.tolist(),
            "tpr": self.tpr.tolist(),
            "std_tpr": self.std_tpr.tolist() if self.std_tpr is not None else None,
            "thresholds": self.thresholds.tolist() if self.thresholds is not None else None,
            "auc": self.auc,
            "std_auc": self.std_auc
        }

    # --------------------------------------------------------

    @classmethod
    def from_dict(cls, d):

        return cls(
            fpr=np.asarray(d["fpr"]),
            tpr=np.asarray(d["tpr"]),
            std_tpr=np.asarray(d['std_tpr']) if d.get('std_tpr', None) is not None else None,
            thresholds=np.asarray(d["thresholds"]),
            auc=d["auc"],
            std_auc= d.get('std_auc', None)
        )


# ============================================================
# ROC PLOTTER
# ============================================================

class ROCPlotter:

    # --------------------------------------------------------

    @staticmethod
    def plot(curves: list[ROCCurve], save_path, plot_cfg: 'PlotConfig', labels=None, colors=None):
        """
        Plot one or several ROC curves.

        Parameters
        ----------
        curves : ROCCurve or list[ROCCurve]
        save_path : string path to save the plot
        roc_cfg : ROCConfig object
        style_config : 
        """

        roc_cfg = plot_cfg.roc_cfg

        if not isinstance(curves, (list, tuple)):
            curves = [curves]

        if labels is None:
            labels = [None] * len(curves)

        if colors is None:

            default = plot_cfg.default_colors

            colors = [
                default[i % len(default)]
                for i in range(len(curves))
            ]

        plt.figure(figsize=roc_cfg.figsize)

        for curve, label, color in zip(curves, labels, colors):

            if label is None:
                label = f"AUC = {curve.auc:.3f}"

            else:
                label = f"{label} (AUC = {curve.auc:.3f}"

                if curve.std_auc is not None and roc_cfg.show_std:

                    label += f" ± {curve.std_auc:.3f}"

                label += ")"

            plt.plot(
                curve.fpr,
                curve.tpr,
                color=color,
                linewidth=roc_cfg.linewidth,
                alpha=roc_cfg.alpha,
                label=label,
            )

            if (curve.std_tpr is not None and roc_cfg.show_std):

                plt.fill_between(
                    curve.fpr,
                    curve.tpr - curve.std_tpr,
                    curve.tpr + curve.std_tpr,
                    color=color,
                    alpha=0.20,
                )

        if roc_cfg.show_diagonal:

            plt.plot(
                [0, 1],
                [0, 1],
                linestyle=roc_cfg.diagonal_linestyle,
                color=roc_cfg.diagonal_color,
                linewidth=1,
            )

        plt.xlim(0, 1)

        plt.ylim(0, 1.05)

        plt.xlabel(
            roc_cfg.xlabel,
            fontsize=plot_cfg.label_fontsize,
            fontfamily=plot_cfg.font_family,
        )

        plt.ylabel(
            roc_cfg.ylabel,
            fontsize=plot_cfg.label_fontsize,
            fontfamily=plot_cfg.font_family,
        )

        plt.title(
            roc_cfg.title,
            fontsize=plot_cfg.title_fontsize,
            fontfamily=plot_cfg.font_family,
            fontweight="bold",
        )
        plt.legend(
            fontsize=plot_cfg.legend_fontsize,
            loc=roc_cfg.legend_loc,
        )

        plt.grid(roc_cfg.grid)

        plt.tick_params(
            labelsize=plot_cfg.tick_fontsize
        )

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=roc_cfg.dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close()

    # --------------------------------------------------------

    @staticmethod
    def compare(bundles, save_path, level, plot_cfg, labels, colors=None):

        if level == "patient":
            curves = [
                b.patient_roc
                for b in bundles
            ]
        else:
            curves = [
                b.sequence_roc
                for b in bundles
            ]

        ROCPlotter.plot(
            curves=curves,
            labels=labels,
            colors=colors,
            save_path=save_path,
            plot_cfg=plot_cfg,
        )

