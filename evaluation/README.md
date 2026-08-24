# Evaluation

The `evaluation` package evaluates binary classification predictions and
produces persisted results and plots. The reference plotting configuration is
`<WORK_PATH>/configs/test_config.yaml`.

## Evaluation flow

When `main.py` is run with `--test`, the model produces test probabilities and
the associated metadata. The evaluator then:

1. Builds a `ResultBundle` from the true labels, positive-class probabilities,
   patient identifiers, views, and clothing conditions.
2. Saves the result bundle in the run's `results/` directory.
3. Optionally generates a ROC curve, a sequence-level dashboard, and grouped
   metric bar charts according to `test_config.yaml`.
4. Averages result bundles when `NUM_RUNS > 1` and can compare multiple model
   configurations.

Run evaluation through the main pipeline with:

```bash
python main.py -c ce --test --restore_dir <experiment-directory>
```

To train and evaluate in one run:

```bash
python main.py -c ce --train --test
```

The evaluation code expects binary labels and a probability for the positive
class. The negative class corresponds to 0 and positive class to 1.

## Python API

The package exposes `Evaluator`, `PlotConfig`, and `ResultBundle`:

```python
from evaluation import Evaluator
from evaluation.result_bundle import ResultBundle
```

For a complete pipeline evaluation, call:

```python
result = Evaluator.evaluate_model(
    conf,
    y_true,
    y_prob,
    patient_ids,
    views,
    clothing,
    eval_epoch,
)
```

Here, `y_true` contains binary ground-truth labels and `y_prob` contains the
predicted probability of the positive class. `patient_ids`, `views`, and
`clothing` provide the grouping metadata used by the result bundle and grouped
plots.

For already-created result bundles, `Evaluator.evaluate_config` averages
several runs, while `Evaluator.compare` compares several configuration-level
result bundles.

## `test_config.yaml`

The file has two top-level sections:

```yaml
plots:
  ...
style:
  ...
```

### Configuration labels

`plots.labels` maps configuration identifiers to the labels shown in plot
legends.

```yaml
plots:
  labels:
    ce: "CE"
    triplet_ce_0.5: "CE+TL 0.5"
```

The keys must match the configuration identifiers passed to `main.py` when
comparing configurations. When a configuration has no label, the identifier is used instead.

### ROC curve

The `plots.roc_curve` section controls ROC curve output.

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables ROC plot generation. | `true` |
| `title` | Plot title. | `""` |
| `figsize` | Matplotlib figure dimensions. | `[10, 8]` |
| `dpi` | Requested output resolution. | `300` |
| `alpha` | Curve transparency. | `0.8` |
| `linewidth` | ROC curve line width. | `3` |
| `show_std` | Shows the standard-deviation band and standard deviation of AUC when available. | `true` |
| `legend_loc` | Legend location. | `"lower right"` |
| `show_diagonal` | Shows the random-classifier diagonal. | `true` |
| `diagonal_color` | Diagonal line color. | `"gray"` |
| `diagonal_linestyle` | Diagonal line style. | `"--"` |
| `xlabel` | X-axis label. | `"False Positive Rate"` |
| `ylabel` | Y-axis label. | `"True Positive Rate"` |
| `grid` | Enables the plot grid. | `true` |

The ROC curve and AUC are computed with scikit-learn from binary labels and
positive-class probabilities.

### Grouped metric bar charts

The `plots.bar_chart_by_group` section controls bar charts grouped by view and
by clothing condition.

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables grouped bar chart generation. | `true` |
| `title` | Chart title. | `""` |
| `dpi` | Output resolution. | `300` |
| `legend_loc` | Legend placement. | `"outside"` |
| `bar_width` | Width of each bar. | `0.25` |
| `show_std` | Shows standard deviations when averaged results contain them. | `true` |
| `metrics` | List of metrics to draw. | See below |

Each item in `metrics` has the following fields:

| Field | Description | Reference |
| --- | --- | --- |
| `key` | Metric attribute to read from the grouped metrics object. | `precision`, `recall`, `f1` |
| `label` | Display label for the metric. | `Precision`, `Recall`, `F1-score` |
| `color` | Bar color, usually a hexadecimal color. | `#53A653` |
| `row` | Subplot row on which the metric is displayed. | `0` |


### Sequence dashboard

```yaml
plots:
  dashboard:
    enabled: true
    title: "=== Sequence-level metrics: test ==="
```

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables the sequence-level dashboard. | `true` |
| `title` | Dashboard title. | `=== Sequence-level metrics: test ===` |

Metrics displayed in the dashboard:
- accuracy, number of errors
- precision, specificity, recall, F1-score
- AUC
- confusion matrix
- information about the configuration of the model

### Global style

The `style` section defines shared visual defaults.

| Parameter | Description | Reference |
| --- | --- | --- |
| `font_family` | Font family used by plot labels and titles. | `sans-serif` |
| `title_fontsize` | Title font size. | `16` |
| `label_fontsize` | Axis-label font size. | `14` |
| `tick_fontsize` | Tick-label font size. | `14` |
| `legend_fontsize` | Legend font size. | `14` |
| `default_colors` | Colors used when a plot does not receive explicit colors. | List of hexadecimal colors |
| `default_linestyles` | Line styles available to plots. | `-`, `--`, `-.`, `:` |
| `line_width` | Shared default line width. | `2` |
| `marker_size` | Shared default marker size. | `8` |
| `image_format` | Format used for generated plot files. | `pdf` or `png` |


## Generated files

Evaluation outputs are written below the active experiment and run directory:

```text
results/
  results.json
  predictions.npz
  seq_roc_curve.<format>
  seq_dashboard.png
  view_metrics.<format>
  clothing_metrics.<format>
```

Some files are generated only when their corresponding plot is enabled. When
results are averaged across runs, standard deviations are included in the
averaged result bundle and can be displayed by the relevant plots.

`results.json` is a `ResultBundle` in JSON format that contains neither the predictions nor the reference values.
The predictions and reference values are stored in the file `predictions.npz`.

