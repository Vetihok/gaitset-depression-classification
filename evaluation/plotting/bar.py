from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np

from ..utils import SubsetBCMetrics
from .plotter import BasePlotter

class BarPlotter:

    @staticmethod
    def plot_by_group(metrics: SubsetBCMetrics, save_path, plot_cfg: "PlotConfig"):
        """
        Plot a bar chart for the specified metrics in plot_cfg.bar_cfg and for each group
        (the keys of metrics dictionary).
        """
        bar_cfg = plot_cfg.bar_cfg

        groups = sorted(metrics.metrics.keys())
        x = np.arange(len(groups))

        # Regroupement des métriques par ligne
        rows = {}
        for cfg in bar_cfg.metrics:
            rows.setdefault(cfg.row, []).append(cfg)

        ordered_rows = sorted(rows.keys())
        n_rows = len(ordered_rows)

        fig, axes = plt.subplots(
            nrows=n_rows,
            figsize=(
                max(
                    6,
                    len(groups)
                       * (max(len(v) for v in rows.values()) 
                       * bar_cfg.bar_width + 0.05),
                ),
                4.5 * n_rows,
            ),
            squeeze=False,
            sharex=True,
        )

        axes = axes.ravel()

        for ax, row in zip(axes, ordered_rows):

            metric_configs = rows[row]
            n_metrics = len(metric_configs)

            width = bar_cfg.bar_width
            offset = (n_metrics - 1) * width / 2

            for i, cfg in enumerate(metric_configs):

                values = [
                    getattr(metrics.metrics[g], cfg.key) * 100
                    for g in groups
                ]

                pos = x - offset + i * width

                kwargs = {}

                if bar_cfg.show_std:
                    kwargs["yerr"] = [
                        getattr(metrics.metrics[g], f"std_{cfg.key}") * 100
                        for g in groups
                    ]
                    kwargs["capsize"] = 4

                ax.bar(
                    pos,
                    values,
                    width,
                    label=cfg.label,
                    color=cfg.color,
                    **kwargs,
                )

            BasePlotter._set_labels(
                ax,
                xlabel="",
                ylabel="Percentage",
                style_cfg=plot_cfg,
            )

            BasePlotter._set_title(
                ax,
                title=bar_cfg.title if n_rows == 1 else f"{bar_cfg.title} ({row})",
                style_cfg=plot_cfg,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(groups, rotation=45, ha="right")

        BasePlotter._finish_plot(
            fig,
            axes,
            save_path,
            plot_cfg,
            dpi=bar_cfg.dpi,
            legend_loc=bar_cfg.legend_loc,
            grid=False,
        )

