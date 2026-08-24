from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class BasePlotter:
    """
    Base class shared by every plot.

    Handles:
        - matplotlib figure creation
        - style
        - default colors
        - saving
        - legends
        - axes formatting
    """

    # ==========================================================
    # FIGURE
    # ==========================================================

    @staticmethod
    def _new_figure(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # ==========================================================
    # COLORS
    # ==========================================================

    @staticmethod
    def _get_colors(n_colors, style_cfg, colors=None):
        """
        Return a list of colors.

        Priority:
            1) user colors
            2) default colors from style config
        """

        if colors is not None:
            return colors

        default = style_cfg.default_colors

        if len(default) == 0:
            default = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        return [
            default[i % len(default)]
            for i in range(n_colors)
        ]

    # ==========================================================
    # LINESTYLES
    # ==========================================================

    @staticmethod
    def _get_linestyles(n, style_cfg, linestyles=None):

        if linestyles is not None:
            return linestyles

        default = style_cfg.default_linestyles

        if len(default) == 0:
            default = ["-"]

        return [
            default[i % len(default)]
            for i in range(n)
        ]

    # ==========================================================
    # TITLES
    # ==========================================================

    @staticmethod
    def _set_title(ax, title, style_cfg):

        ax.set_title(
            title,
            fontsize=style_cfg.title_fontsize,
            fontfamily=style_cfg.font_family,
            fontweight="bold",
        )

    # ==========================================================
    # LABELS
    # ==========================================================

    @staticmethod
    def _set_labels(ax, xlabel, ylabel, style_cfg):

        ax.set_xlabel(
            xlabel,
            fontsize=style_cfg.label_fontsize,
            fontfamily=style_cfg.font_family,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=style_cfg.label_fontsize,
            fontfamily=style_cfg.font_family,
        )

    # ==========================================================
    # AXES STYLE
    # ==========================================================

    @staticmethod
    def _apply_axes_style(
        ax,
        style_cfg,
        grid=True,
        xlim=None,
        ylim=None,
    ):

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.tick_params(
            labelsize=style_cfg.tick_fontsize
        )

        if grid:
            ax.grid(
                True,
                alpha=0.3,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ==========================================================
    # LEGEND
    # ==========================================================

    @staticmethod
    def _legend(ax, style_cfg, location="best", outside=False):

        handles, labels = ax.get_legend_handles_labels()

        if not handles:
            return

        kwargs = dict(
            fontsize=style_cfg.legend_fontsize,
            frameon=False,
        )

        if outside:
            kwargs["loc"] = "upper left"
            kwargs["bbox_to_anchor"] = (1.02, 1)
        else:
            kwargs["loc"] = location

        ax.legend(handles, labels, **kwargs)

    # ==========================================================
    # DIAGONAL
    # ==========================================================

    @staticmethod
    def _draw_diagonal(
        ax,
        color="gray",
        linestyle="--",
    ):

        ax.plot(
            [0, 1],
            [0, 1],
            color=color,
            linestyle=linestyle,
            linewidth=1,
            alpha=0.5,
        )

    # ==========================================================
    # FILL STD
    # ==========================================================

    @staticmethod
    def _plot_std_band(
        ax,
        x,
        mean,
        std,
        color,
        alpha=0.20,
    ):

        if std is None:
            return

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=alpha,
        )

    # ==========================================================
    # SAVE
    # ==========================================================

    @staticmethod
    def _save(fig, save_path, dpi):

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)

    # ==========================================================
    # COMPLETE
    # ==========================================================

    @classmethod
    def _finish_plot(
        cls, fig, ax, save_path, style_cfg, dpi, legend_loc="best", grid=True,
        xlim=None, ylim=None,
    ):

        axes = np.atleast_1d(ax)

        for a in axes:

            cls._apply_axes_style(
                a,
                style_cfg,
                grid=grid,
                xlim=xlim,
                ylim=ylim,
            )

            cls._legend(
                a,
                style_cfg,
                legend_loc if not legend_loc == 'outside' else 'best',
                outside=(legend_loc == 'outside')
            )

        fig.tight_layout()

        cls._save(
            fig,
            save_path,
            dpi,
        )