from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class MetricConfig:
    key: str          # Nom de l'attribut dans SubsetBCMetrics (ex: "f1", "accuracy")
    label: str        # Légende
    color: str        # Couleur hex (#RRGGBB)
    row: int          # Sous-graphique sur lequel affiché la barre

@dataclass
class BarConfig:

    enabled: bool = True
    title: str = "Per-view Metrics"
    dpi: int = 300
    legend_loc: str = "top right"
    show_std: bool = False

    bar_width: float = 0.25

    metrics: List[MetricConfig] = None  # Liste des métriques à afficher

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BarConfig':
        """
        Create a new BarConfig object from a dictionnary.

        Args:
            data: Dictionnary with the following possible parameters:
                  enabled, title, dpi, legend_loc, show_std, bar_width, metrics.

        Returns:
            BarConfig: configured instance.
        """
        metrics = None
        if 'metrics' in data and data['metrics']:
            metrics = [MetricConfig(**m) for m in data['metrics']]

        return cls(
            enabled=data.get('enabled', True),
            title=data.get('title', "Per-view Metrics"),
            dpi=data.get('dpi', 300),
            legend_loc=data.get('legend_loc', "top right"),
            show_std=data.get('show_std', False),
            bar_width=data.get('bar_width', 0.25),
            metrics=metrics
        )