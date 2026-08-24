import json
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np

from .utils import BClassificationMetrics
from .utils import SubsetBCMetrics
from .utils import Aggregator
from .plotting import ROCCurve

log = logging.getLogger(__name__)

@dataclass
class ResultBundle:
    """
    Complete evaluation results for one trained model.

    This class stores the raw predictions together with all computed metrics.
    Metrics can always be recomputed from the raw predictions.
    """
    eval_epoch: int

    # ------------------------------------------------------------------
    # Raw predictions
    # ------------------------------------------------------------------

    y_true: np.ndarray
    y_prob: np.ndarray

    patient_ids: np.ndarray
    views: np.ndarray
    clothing: np.ndarray

    threshold: float = 0.5

    # ------------------------------------------------------------------
    # Computed metrics
    # ------------------------------------------------------------------

    sequence_roc: ROCCurve | None = None
    patient_roc: ROCCurve | None = None

    sequence_metrics: BClassificationMetrics | None = None
    patient_metrics: BClassificationMetrics | None = None

    view_metrics: SubsetBCMetrics | None = None
    clothing_metrics: SubsetBCMetrics | None = None


    # ==================================================================
    # Construction
    # ==================================================================

    @classmethod
    def from_pred(cls, y_true, y_prob, patient_ids, views, clothing, eval_epoch, threshold=0.5):

        bundle = cls(
            y_true=np.asarray(y_true),
            y_prob=np.asarray(y_prob),
            patient_ids=np.asarray(patient_ids),
            views=np.asarray(views),
            clothing=np.asarray(clothing),
            eval_epoch=eval_epoch,
            threshold=threshold,
        )

        bundle.compute()

        return bundle

    @classmethod
    def average(cls, bundles: list["ResultBundle"]):
        
        return cls(
            eval_epoch=0,
            y_true=None,
            y_prob=None,
            patient_ids=None,
            views=None,
            clothing=None,
            sequence_metrics=BClassificationMetrics.average(
                [b.sequence_metrics for b in bundles]
            ),

            patient_metrics=BClassificationMetrics.average(
                [b.patient_metrics for b in bundles]
            ),

            sequence_roc=ROCCurve.average(
                [b.sequence_roc for b in bundles]
            ),

            patient_roc=ROCCurve.average(
                [b.patient_roc for b in bundles]
            ),

            view_metrics=SubsetBCMetrics.average(
                [b.view_metrics for b in bundles]
            ),

            clothing_metrics=SubsetBCMetrics.average(
                [b.clothing_metrics for b in bundles]
            ),
        )
    
    # ==================================================================
    # Metrics computation
    # ==================================================================

    def compute(self):

        self.sequence_metrics = BClassificationMetrics.from_pred(
            self.y_true,
            self.y_prob,
            self.threshold,
        )

        patient_y, patient_prob, _ = Aggregator.aggregate(
            self.y_true,
            self.y_prob,
            self.patient_ids,
        )

        self.patient_metrics = BClassificationMetrics.from_pred(
            patient_y,
            patient_prob,
            self.threshold,
        )

        self.view_metrics = SubsetBCMetrics.from_pred(
            self.y_true,
            self.y_prob,
            self.views,
            self.threshold,
        )

        self.clothing_metrics = SubsetBCMetrics.from_pred(
            self.y_true,
            self.y_prob,
            self.clothing,
            self.threshold,
        )

        self.sequence_roc = ROCCurve.from_pred(
            self.y_true,
            self.y_prob
        )

        self.patient_roc = ROCCurve.from_pred(
            patient_y,
            patient_prob
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def save(self, save_dir, save_predictions=True, include_std=False):

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if save_predictions:
            self._save_predictions(
                save_dir / "predictions.npz"
            )

        with open(save_dir / "results.json", "w") as f:
            json.dump(
                self.to_dict(include_std=include_std),
                f,
                indent=2
            )

    def _save_predictions(self, filename):
        
        np.savez_compressed(
            filename,
            y_true=self.y_true,
            y_prob=self.y_prob,
            patient_ids=self.patient_ids,
            views=self.views,
            clothing=self.clothing,
        )

    # ==================================================================
    # Loading
    # ==================================================================

    @classmethod
    def load(cls, save_dir, load_predictions=True):
        """
        Return the loaded ResultBundle.

        Args:
            save_dir: directory to find predictions.npz and/or results.json
            load_predictions: if ```True```, ResultBundle is loaded from predictions.npz and results.json, and all the data contained in results.json is computed again. 
                              Otherwise, it is only loaded from results.json and arrays stored in predictions.npz are initialised to ```None```. 
                              Default: ```True``` 

        """
        save_dir = Path(save_dir)

        with open(save_dir / "results.json") as f:
            data = json.load(f)

        if load_predictions:
            arrays = np.load(save_dir / "predictions.npz")

            bundle = cls.from_pred(
                y_true=arrays["y_true"],
                y_prob=arrays["y_prob"],
                patient_ids=arrays["patient_ids"],
                views=arrays["views"],
                clothing=arrays["clothing"],
                eval_epoch=data["eval_epoch"],
                threshold=data["threshold"],
            )
        else:
            bundle = cls(
                y_true=None,
                y_prob=None,
                patient_ids=None,
                views=None,
                clothing=None,
                eval_epoch=0,
                threshold=0.,
                sequence_metrics=data['sequence_metrics'],
                sequence_roc=ROCCurve.from_dict(data['sequence_roc']),
                clothing_metrics=data['clothing_metrics'],
                view_metrics=data['view_metrics'],
                patient_metrics=data['patient_metrics'],
                patient_roc=ROCCurve.from_dict(data['patient_roc']),
            )

        return bundle

    # ==================================================================
    # Utilities
    # ==================================================================

    def to_dict(self, include_std=False):
        return {
            "eval_epoch": self.eval_epoch,
            "threshold": self.threshold,
            "sequence_metrics": self.sequence_metrics.to_dict(include_std),
            "patient_metrics": self.patient_metrics.to_dict(include_std),
            "sequence_roc": self.sequence_roc.to_dict(),
            "patient_roc": self.patient_roc.to_dict(),
            "view_metrics": {
                k: v.to_dict(include_std)
                for k, v in self.view_metrics.metrics.items()
            },
            "clothing_metrics": {
                k: v.to_dict(include_std)
                for k, v in self.clothing_metrics.metrics.items()
            },
        }

    