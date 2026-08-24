from dataclasses import dataclass, field
import yaml

from .plotting.roc_config import ROCConfig
from .plotting.bar_config import BarConfig
from config import _load_yaml_file
from env_manager import EnvManager

@dataclass
class PlotConfig:

    # -----------------------------
    # Plots
    # -----------------------------

    roc_cfg: ROCConfig
    bar_cfg: BarConfig
    dashboard_cfg: dict
    labels: dict = field(default_factory=dict)

    # -----------------------------
    # Global style
    # -----------------------------

    font_family: str = "sans-serif"

    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    legend_fontsize: int = 12

    default_colors: list = field(default_factory=list)
    default_linestyles: list = field(default_factory=list)

    line_width: float = 2
    marker_size: int = 8

    image_format: str = "pdf"



    # -----------------------------
    # Precision Recall
    # -----------------------------

    precision_recall_curve: dict = field(default_factory=dict)

    # -----------------------------
    # Radar
    # -----------------------------

    radar_chart: dict = field(default_factory=dict)

    # -----------------------------
    # Metrics bars
    # -----------------------------

    metrics_bar_chart: dict = field(default_factory=dict)

    # -----------------------------
    # Summary table
    # -----------------------------

    summary_table: dict = field(default_factory=dict)

    # -----------------------------
    # View plots
    # -----------------------------

    view_metrics_bar_chart: dict = field(default_factory=dict)

    # =========================================================

    @classmethod
    def from_yaml(cls, yaml_file: str):
        """
        Import yaml_file

        Args:
            yaml_file: path to eval config file
        """
        cfg = _load_yaml_file(yaml_file)

        style = cfg.get("style", {})
        plots = cfg.get("plots", {})

        labels = plots.get("labels", {})

        for config in EnvManager.get_instance().config_list:
            if config not in labels:
                labels[config] = config

        return cls(

            font_family=style.get("font_family", "sans-serif"),

            title_fontsize=style.get("title_fontsize", 16),
            label_fontsize=style.get("label_fontsize", 14),
            tick_fontsize=style.get("tick_fontsize", 12),
            legend_fontsize=style.get("legend_fontsize", 12),

            default_colors=style.get("default_colors", []),
            default_linestyles=style.get("default_linestyles", []),

            line_width=style.get("line_width", 2),
            marker_size=style.get("marker_size", 8),

            image_format=style.get("image_format", "pdf"),

            roc_cfg=ROCConfig.from_dict(plots.get("roc_curve", {})),
            bar_cfg=BarConfig.from_dict(plots.get("bar_chart_by_group", {})),
            dashboard_cfg=plots.get("dashboard", {}),
            labels=labels,
        )