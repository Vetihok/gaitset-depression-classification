# GaitSet for Depression Classification

Research project using gait sequences to classify depression. The repository
uses GaitSet to extract sequence representations, followed by a binary
classification model evaluated from predicted probabilities.

This project is based on the original [GaitSet repository](https://github.com/AbnerHqC/GaitSet).

The pipeline can train a model, evaluate an existing model, repeat a
configuration several times, aggregate results, and compare multiple
configurations.

## Repository structure

- `main.py`: entry point for the training and evaluation pipeline.
- `env_config.yaml`: runtime and experiment settings.
- `config/`: loading and management of YAML model configurations.
- `model/`: initialization and implementation of the trained model.
- `evaluation/`: metric computation, result saving, and visualization
  generation.
- `preprocessing/`: data preparation.

## Configuration

Configuration files are read from the `configs` directory under the
`WORK_PATH` defined in `env_config.yaml`:

```text
<WORK_PATH>/configs/data_config.yaml
<WORK_PATH>/configs/config_<id>.yaml
<WORK_PATH>/configs/test_config.yaml
```

For further information on `env_config.yaml`, refer to the [Runtime Environment](#runtime-environment) section.

`data_config.yaml` describes data shared by experiments, `config_<id>.yaml`
describes a model configuration, and `test_config.yaml` controls evaluation
outputs. The latter two formats are documented in dedicated README files:

- [Configure a model with YAML](config/README.md)
- [Evaluate models and configure tests](evaluation/README.md)

## Usage

From the repository root, run a configuration with:

```bash
python main.py -c <id> --train --test
```

For example, to train and then evaluate `config_1.yaml`:

```bash
python main.py -c 1 --train --test
```

Multiple configurations can be run in the same experiment:

```bash
python main.py -c 1 2 3 --train --test
```

Main options:
- `-c`, `--config`: list of configurations identifier to train or test. Identifiers can be any string of characters.
- `--train`: train the model and save its checkpoints.
- `--test`: extract test embeddings and compute evaluation results.
- `--restore_dir <path>`: resume an existing experiment.
- `--debug`: write outputs to a `debug` directory.
- `--summary`: display the model summary.
- `--log_level DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL`: set the logging level.

The `--train` and `--test` options can also explicitly receive `TRUE` or
`FALSE`.

## Results and repetitions

Outputs are organized under `WORK_PATH/exp` (or `WORK_PATH/debug` in debug
mode), optionally under `SUBFOLDER`, and then under an experiment directory.
When multiple configurations or repetitions are used, `config_<id>` and
`run_<id>` subdirectories are created. The results of each run are saved in
`results/`, including `results.json` and `predictions.npz`; checkpoints are
saved in `checkpoint/`.

When `NUM_RUNS` is greater than 1, the pipeline also computes average results
and standard deviations. When multiple configurations are run, their results
can be compared within the same experiment.

## Runtime environment

The `env_config.yaml` file defines the working path, CPU/GPU selection, visible
GPUs, number of repetitions, and behavior when restoring an experiment. At a
minimum, adapt `WORK_PATH` to your machine before starting an experiment.

The repository does not currently contain a dependency file or a centralized
installation procedure. The Python environment must provide the dependencies
used by the code, including PyTorch, NumPy, scikit-learn, PyYAML, Matplotlib,
and pytz.