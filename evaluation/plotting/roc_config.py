from dataclasses import dataclass


@dataclass
class ROCConfig:
    """
    Configuration of ROC curve plots.
    """

    enabled: bool = True
    title: str = "ROC Curve"
    figsize: tuple[int, int] = (10, 8)
    dpi: int = 300
    alpha: float = 0.8
    linewidth: float = 2.0
    show_std: bool = False
    legend_loc: str = "lower right"

    show_diagonal: bool = True
    diagonal_color: str = "gray"
    diagonal_linestyle: str = "--"

    xlabel: str = "False Positive Rate"
    ylabel: str = "True Positive Rate"

    grid: bool = False

    @classmethod
    def from_dict(cls, roc_dict):
        """
        Create a new ROCConfig object from a dictionnary.

        Args:
            roc_dict: Dictionnary with the following possible parameters:
                    enabled, title, figsize, dpi, alpha, linewidth, legend_loc, 
                    show_diagonal, diagonal_color, diagonal_linestyle, xlabel, ylabel, grid.

        Returns:
            BarConfig: configured instance.
        """
        return cls(
            enabled=roc_dict.get("enabled", True),
            title=roc_dict.get("title", "ROC Curve"),
            figsize=roc_dict.get("figsize", (10, 8)),
            dpi=roc_dict.get("dpi", 300),
            alpha=roc_dict.get("alpha", 0.8),
            linewidth=roc_dict.get("linewidth", 0.2),
            show_std=roc_dict.get("show_std", False),
            legend_loc=roc_dict.get("legend_loc", "best"),
            show_diagonal=roc_dict.get("show_diagonal", True),
            diagonal_color=roc_dict.get("diagonal_color", "gray"),
            diagonal_linestyle=roc_dict.get("diagonal_linestyle", "--"),
            xlabel=roc_dict.get("xlabel", "False Positive Rate"),
            ylabel=roc_dict.get("ylabel", "True Positive Rate"),
            grid=roc_dict.get("grid", False),
        )
