import logging
import os
from typing import List

from .test_config import PlotConfig
from .result_bundle import ResultBundle
from .plotting import ROCPlotter, DashboardPlotter, BarPlotter
from env_manager import EnvManager

log = logging.getLogger(__name__)

class Evaluator:
    """Evaluate binary classification results and generate evaluation plots.

    The evaluator relies on the active :class:`EnvManager` instance to locate
    ``test_config.yaml`` and to save result bundles and generated figures.
    """

    @staticmethod
    def evaluate_model(conf, y_true, y_prob, patient_ids, views, clothing, eval_epoch, save_suffix=""):
        """Evaluate predictions from one model run.

        A :class:`ResultBundle` is created from the predictions and metadata,
        saved to the active results directory, and used to generate the
        enabled ROC, dashboard, and grouped bar plots.

        Parameters
        ----------
        conf : dict
            Complete model and data configuration used to produce the
            predictions. It is passed to the dashboard plotter.
        y_true : array-like
            Ground-truth binary labels.
        y_prob : array-like
            Predicted probabilities for the positive class.
        patient_ids : array-like
            Patient identifier associated with each prediction.
        views : array-like
            View identifier associated with each prediction.
        clothing : array-like
            Clothing-condition identifier associated with each prediction.
        eval_epoch : int
            Epoch of the model used for evaluation.
        save_suffix : str, optional
            Suffix appended to generated plot filenames.

        Returns
        -------
        ResultBundle
            The result bundle created from the predictions.

        Raises
        ------
        RuntimeError
            If the environment manager has not been initialized.
        FileNotFoundError
            If the active work directory does not contain ``test_config.yaml``.
        """
        env = EnvManager.get_instance()
        config_file = os.path.join(env.get_configs_dir(), f"test_config.yaml")
        plot_cfg = PlotConfig.from_yaml(config_file)

        result_bundle = ResultBundle.from_pred(y_true, y_prob, patient_ids, views, clothing, eval_epoch)

        result_bundle.save(env.get_dir(results=True))

        if plot_cfg.roc_cfg.enabled:
            ROCPlotter.plot(result_bundle.sequence_roc, 
                            env.get_general_save_path("seq_roc_curve" + save_suffix, plot_cfg.image_format, results=True), 
                            plot_cfg)

        if plot_cfg.dashboard_cfg['enabled']:
            DashboardPlotter.plot_metrics(result_bundle.sequence_metrics, 
                              env.get_general_save_path("seq_dashboard" + save_suffix, "png", results=True),
                              conf, title=plot_cfg.dashboard_cfg['title'], eval_epoch=eval_epoch)

        if plot_cfg.bar_cfg.enabled:
            BarPlotter.plot_by_group(result_bundle.view_metrics,
                                env.get_general_save_path("view_metrics" + save_suffix, plot_cfg.image_format, results=True),
                                plot_cfg)
            BarPlotter.plot_by_group(result_bundle.clothing_metrics,
                                env.get_general_save_path("clothing_metrics" + save_suffix, plot_cfg.image_format, results=True),
                                plot_cfg)

        return result_bundle

    @staticmethod
    def evaluate_config(conf, result_bundles: List[ResultBundle], save_suffix="") -> ResultBundle:
        """Aggregate and evaluate the runs of one configuration.

        The input result bundles are averaged, saved without individual
        predictions, and used to generate the enabled aggregate plots.

        Parameters
        ----------
        conf : dict
            Complete model and data configuration. It is accepted for API
            consistency; aggregate plotting currently uses the plot
            configuration and the result bundles.
        result_bundles : list of ResultBundle
            Result bundles produced by the individual runs of one model
            configuration.
        save_suffix : str, optional
            Suffix appended to generated plot filenames.

        Returns
        -------
        ResultBundle
            The averaged result bundle, including standard deviations when
            supported by the result bundle implementation.

        Raises
        ------
        RuntimeError
            If the environment manager has not been initialized.
        ValueError
            If ``result_bundles`` is empty.
        """
        env = EnvManager.get_instance()

        res = ResultBundle.average(result_bundles)
        res.save(env.get_dir(results=True), save_predictions=False, include_std=True)

        config_file = os.path.join(env.get_configs_dir(), f"test_config.yaml")
        plot_cfg = PlotConfig.from_yaml(config_file)

        if plot_cfg.roc_cfg.enabled:
            ROCPlotter.plot(res.sequence_roc, 
                            env.get_general_save_path("seq_roc_curve" + save_suffix, plot_cfg.image_format, results=True), 
                            plot_cfg,
                            labels=[plot_cfg.labels[env.current_config_id]],)


        if plot_cfg.bar_cfg.enabled:
            BarPlotter.plot_by_group(res.view_metrics,
                                env.get_general_save_path("view_metrics" + save_suffix, plot_cfg.image_format, results=True),
                                plot_cfg)
            BarPlotter.plot_by_group(res.clothing_metrics,
                                env.get_general_save_path("clothing_metrics" + save_suffix, plot_cfg.image_format, results=True),
                                plot_cfg)

        return res

    @staticmethod
    def compare(result_bundles: List[ResultBundle], save_suffix=""):
        """Compare result bundles from multiple configurations.

        The method loads the active plot configuration and, when ROC plotting
        is enabled, draws the sequence-level ROC curves in one comparison
        figure. Labels are taken from ``plots.labels`` in ``test_config.yaml``.

        Parameters
        ----------
        result_bundles : list of ResultBundle
            Result bundles to compare. The order must match the labels defined
            in the evaluation configuration.
        save_suffix : str, optional
            Suffix appended to the generated comparison plot filename.

        Raises
        ------
        RuntimeError
            If the environment manager has not been initialized.
        FileNotFoundError
            If the active work directory does not contain ``test_config.yaml``.
        """
        env = EnvManager.get_instance()
        
        config_file = os.path.join(env.get_configs_dir(), f"test_config.yaml")
        plot_cfg = PlotConfig.from_yaml(config_file)

        if plot_cfg.roc_cfg.enabled:
            labels = [v for k, v in plot_cfg.labels.items()]
            log.debug(f'{labels}')
            ROCPlotter.plot([res.sequence_roc for res in result_bundles], 
                            env.get_general_save_path("seq_roc_curve_comparison" + save_suffix, plot_cfg.image_format, results=True), 
                            plot_cfg, labels=labels)