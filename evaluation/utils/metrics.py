from dataclasses import dataclass
import json
import numpy as np
from sklearn.metrics import roc_auc_score

@dataclass
class BClassificationMetrics:

    accuracy: float = 0.
    precision: float = 0.
    recall: float = 0.
    specificity: float = 0.
    f1: float = 0.
    auc: float = 0.

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    threshold: float = 0.5

    std_accuracy: float = 0.
    std_precision: float = 0.
    std_recall: float = 0.
    std_specificity: float = 0.
    std_f1: float = 0.
    std_auc: float = 0.

    std_tp: float = 0.
    std_tn: float = 0.
    std_fp: float = 0.
    std_fn: float = 0.

    @classmethod
    def from_pred(cls, y_true, y_prob, threshold=0.5):
        y_true = np.asarray(y_true).astype('int32')
        y_pred = (np.asarray(y_prob) >= threshold).astype('int32')
    
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
    
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        auc = roc_auc_score(y_true, y_prob)

        return cls(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1=f1,
            auc=auc,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            threshold=threshold,
        )

    @classmethod
    def average(cls, metrics_list):
        """
        Average several BClassificationMetrics.
        """

        if len(metrics_list) == 0:
            raise ValueError("metrics_list is empty.")

        result = cls()

        attributes = [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

        for attr in attributes:

            values = np.asarray(
                [getattr(m, attr) for m in metrics_list],
                dtype=float,
            )

            setattr(result, attr, values.mean())
            setattr(result, f"std_{attr}", values.std())

        result.threshold = metrics_list[0].threshold
        result.eval_epoch = None

        return result

    def to_dict(self, include_std=False):
        dic = {}
        attributes = [
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auc",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

        for attr in attributes:

            dic[attr] = getattr(self, attr)

            if include_std:
                dic[f"std_{attr}"] = getattr(self, f"std_{attr}")

        return dic

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

    def __str__(self):
        return (
            f"Accuracy    : {self.accuracy:.3f}\n"
            f"Precision   : {self.precision:.3f}\n"
            f"Recall      : {self.recall:.3f}\n"
            f"Specificity : {self.specificity:.3f}\n"
            f"F1          : {self.f1:.3f}\n"
            f"AUC         : {self.auc:.3f}"
        )

    def save_json(self, save_path, include_std=False):
        with open(save_path, 'w') as f:
            json.dump(self.to_dict(include_std=include_std), f, indent=2)

    @classmethod
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            return cls.from_dict(json.load(f))

    