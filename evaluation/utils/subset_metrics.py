from dataclasses import dataclass
import json
from .metrics import BClassificationMetrics

@dataclass
class SubsetBCMetrics:
    """
    Evaluation results for several subgroups.
    """
    metrics: dict[str, BClassificationMetrics]

    @classmethod
    def from_pred(cls, y_true, y_prob, groups, threshold=0.5):
        results = {}

        for group in set(groups):
            mask = groups == group

            results[group] = BClassificationMetrics.from_pred(y_true[mask], y_prob[mask], threshold=threshold)
        
        return cls(results)

    @classmethod
    def average(cls, subsets: list["SubsetBCMetrics"]):
        """
        Average several grouped metrics.

        Parameters
        ----------
        subsets : list[dict]

        Returns
        -------
        dict
        """

        if len(subsets) == 0:
            raise ValueError("subsets is empty.")

        result = {}

        groups = subsets[0].metrics.keys()

        for group in groups:

            metrics = [
                subset.metrics[group]
                for subset in subsets
            ]

            result[group] = BClassificationMetrics.average(metrics)

        return SubsetBCMetrics(result)


    def save_json(self, save_path):
        with open(save_path, 'w') as f:
            json.dump({group: metrics.to_dict() for group, metrics in self.metrics.items()}, f, indent=2)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return cls({group: BClassificationMetrics.from_dict(metrics_dict) for group, metrics_dict in data.items()})