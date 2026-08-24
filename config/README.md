# Model Configuration

This directory contains the Python code used to load and expose YAML
configuration files. The model and data YAML files themselves are stored under
`<WORK_PATH>/configs/`.

The reference files for this documentation are:

- `data_config.yaml`: shared dataset and partitioning settings.
- `config_ce.yaml`: reference model configuration.

The main script loads `data_config.yaml`, loads the selected
`config_<id>.yaml`, merges both dictionaries, and stores the result in the
global configuration object used by the model.

## File naming and loading

Model configurations must follow this naming convention:

```text
<WORK_PATH>/configs/config_<id>.yaml
```

Run a configuration from the repository root with:

```bash
python main.py -c <id> --train --test
```

For example:

```bash
python main.py -c ce --train --test
```

The YAML loader supports the `!tuple` tag used by the reference files:

```yaml
split: !tuple [0.70, 0.10, 0.20]
batch_size: !tuple [4, 16]
```

## Data configuration

The data configuration is expected to have a top-level `data` section.

| Parameter | Description | Reference |
| --- | --- | --- |
| `dataset_path` | Path to the preprocessed dataset. | `/home/docker/DATA/D-Gait/D-Gait-Silhouette-preprocess` |
| `resolution` | Spatial resolution used by the input silhouettes. | `64` |
| `dataset` | Dataset identifier selected by the data-loading pipeline. | `D-Gait` |
| `cache` | Whether dataset data should be cached. | `true` |
| `load_all_train_data` | Whether all training data should be loaded at initialization. | `false` |
| `load_all_test_data` | Whether all test data should be loaded at initialization. | `false` |
| `partitioning.split` | Proportions assigned to train, validation, and test, in that order. The reference uses 70%, 10%, and 20%. | `!tuple [0.70, 0.10, 0.20]` |
| `partitioning.seed` | Seed used for partitioning. | `42` |
| `partitioning.split_mode` | Partitioning strategy. The reference lists `random`, `weighted_random`, `subject`, and `sequence` as available choices. (See `load_data` in `model/utils/data_loader.py`) | `weighted_random` |


## Model configuration

All model settings are nested under `model`.

### General training settings

| Parameter | Description | Reference |
| --- | --- | --- |
| `model_name` | Name used to identify the model configuration and outputs. | `D-GaitSet` |
| `save_summary` | Whether the model summary is saved when requested by the pipeline. | `false` |
| `num_epochs` | Maximum number of training epochs. | `50` |
| `restore_epoch` | Checkpoint epoch to restore. `0` starts a new run; `-1` asks the restore logic to select a checkpoint according to the restore mode. | `0` |
| `eval_interval` | Number of epochs between validation evaluations. | `1` |
| `hidden_dim` | Hidden feature dimension produced by the GaitSet encoder and consumed by the classifier head. | `256` |
| `num_workers` | Number of data-loader worker processes. | `6` |
| `frame_num` | Number of frames used for a gait sequence. | `30` |
| `batch_size` | Training batch-size tuple. | `!tuple [4, 16]` |
| `embeddings_batch_size` | Batch size used while extracting test embeddings. | `64` |

tuple `batch_size` is only useful with `TripletLoss`: 
- the first value is for the number of subject
- the second value is for the number of sequences per subject

### Pretrained model

```yaml
pretrained_model:
  enabled: true
  load_state_dict_strict: false
  PRETRAINED_PATH: /path/to/checkpoint.ptm
```

| Parameter | Description |
| --- | --- |
| `enabled` | Enables loading weights from a pretrained model. |
| `load_state_dict_strict` | Controls whether the state dictionary must match strictly. `false` allows non-strict loading. |
| `PRETRAINED_PATH` | Path to the pretrained checkpoint file. |

This only loads weights for GaitSet, not for the classification head.

### Sampler

```yaml
sampler:
  type: CETripletSampler
  sample_type: all
  replacement: false
  weight_damping_factor: 0.5
```

| Parameter | Description | Available values or reference |
| --- | --- | --- |
| `type` | Sampler implementation used by the training data loader. | `ClassificationSampler`, `WeightedRandomSampler`, `SequentialSampler`, `RandomSampler`, `CETripletSampler` |
| `sample_type` | Sampling mode used in the collation step. (See `collate_fn` in ``model/model.py``) | `random`, `all` |
| `replacement` | Whether weighted sampling can select an item more than once. | `false` |
| `weight_damping_factor` | Damping factor applied to sampling weights. | `0.5`; default noted in the YAML is `1.0` |

`replacement` and `weight_damping_factor` are primarily relevant to weighted
sampling.

### Losses

The enabled losses are combined according to their configured weights.

#### Triplet loss

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables the triplet-loss term. | `false` |
| `weight` | Relative weight of the triplet-loss term. | `4.0` |
| `margin` | Margin used by the triplet loss. | `0.2` |
| `hard_or_full` | Selects the triplet-loss variant: `full` or `hard`. | `full` |


#### Cross-entropy loss

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables cross-entropy loss. | `true` |
| `weight` | Relative weight of the cross-entropy term. | `1.0` |
| `label_smoothing` | Label-smoothing value passed to the loss. | `0.0` |
| `class_weights` | Optional weight for each class, in class-index order. `null` disables weighting. | `null` |
| `reduction` | Reduction applied to the per-sample loss. | `none`, `mean`, `sum`; reference: `mean` |

#### Focal loss

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables focal loss. | `false` |
| `weight` | Relative weight of the focal-loss term. | `1.0` |
| `alpha` | Focal-loss balancing factor. | `0.25` |
| `gamma` | Focusing parameter that controls how strongly easy examples are down-weighted. | `2.0` |
| `class_weights` | Optional weight for each class. | `null` |
| `reduction` | Reduction applied to the per-sample loss. | `none`, `mean`, `sum`; reference: `mean` |

#### Loss formula

Each loss is computed only if enabled. The final loss is the following:

`loss` = `triplet_loss.weight` * `triplet_loss` + `cross_entropy_loss.weight` * `cross_entropy_loss` + `focal_loss.weight` * `focal_loss`

### Optimizer

| Parameter | Description | Reference |
| --- | --- | --- |
| `type` | Optimizer implementation. | `Adam` or `SGD`; reference: `Adam` |
| `encoder_lr` | Learning rate for encoder parameters. | `0.0001` |
| `classifier_lr` | Learning rate for classifier parameters. | `0.0001` |
| `weight_decay` | Weight-decay coefficient. | `0.0001` |


### Learning-rate scheduler

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables learning-rate scheduling. | `false` |
| `type` | Scheduler implementation. | `CosineAnnealingLR`, `OneCycleLR`, `PolynomialLR`, `MultiStepLR` |
| `eta_min` | Minimum learning rate for `CosineAnnealingLR`. | `0` |
| `max_lr` | Maximum learning rate for `OneCycleLR`. | `0.01` |
| `pct_start` | Fraction of the cycle spent increasing the learning rate for `OneCycleLR`. | `0.3` |
| `power` | Polynomial power for `PolynomialLR`. | `1.0` |
| `min_lr` | Minimum learning rate for `PolynomialLR`. | `0.0` |
| `milestones` | Epochs at which `MultiStepLR` changes the learning rate. | `[15, 25, 35]` |
| `gamma` | Multiplicative factor applied by `MultiStepLR` at each milestone. | `0.1` |

### Regularization

#### R-Drop

| Parameter | Description | Reference |
| --- | --- | --- |
| `r_drop.enabled` | Enables R-Drop regularization. | `false` |
| `r_drop.alpha` | Weight of the R-Drop regularization term. | `10.0` |

Implemented as described in the [paper by Huang et al.](https://ieeexplore.ieee.org/document/11343509)

#### Dropout

| Parameter | Description | Reference |
| --- | --- | --- |
| `dropout.enabled` | Enables encoder and classifier dropout. | `true` |
| `dropout.encoder_p` | Dropout probability in the encoder. | `0.15` |
| `dropout.head_p` | Default dropout probability for the classifier head. | `0.3` |

### Classifier head

An empty `layers` list creates a single linear classifier from `hidden_dim` to
2 output classes:

```yaml
classifier_head:
  layers: []
```

Custom heads are defined as an ordered list. Supported layer types implemented
by the classifier are `linear`, `relu`, `leaky_relu`, `dropout`, `sigmoid`, and
`tanh`.

```yaml
classifier_head:
  layers:
    - type: linear
      out_features: 256
    - type: relu
    - type: dropout
      p: 0.3
    - type: linear
      out_features: 2
```

| Layer type | Parameters |
| --- | --- |
| `linear` | `out_features` |
| `relu` | none |
| `leaky_relu` | `negative_slope` (default `0.1`) |
| `dropout` | `p`; defaults to `dropout.head_p` |
| `sigmoid` | none |
| `tanh` | none |

The final layer should produce two outputs for binary classification.

### Freezing encoder layers

| Parameter | Description | Reference |
| --- | --- | --- |
| `freeze.enabled` | Enables layer freezing. | `false` |
| `freeze.freeze_whole_encoder` | Freezes the complete encoder when enabled. | `false` |
| `freeze.unfreeze_epoch` | Epoch at which frozen layers are unfrozen; `-1` disables scheduled unfreezing. | `-1` |
| `freeze.encoder_layers` | Per-layer flags for the encoder blocks and pooling modules. | See reference YAML |
| `freeze.freeze_classifier` | Freezes classifier parameters. This option is currently marked as not implemented in the reference configuration. | `false` |

The `encoder_layers` keys correspond to the GaitSet modules `set_layer1` to
`set_layer6`, `gl_layer1` to `gl_layer4`, `gl_hpm`, and `x_hpm`. If a layer parameter is set to `true`, it will be frozen during training.

### Early stopping

| Parameter | Description | Reference |
| --- | --- | --- |
| `early_stop.enabled` | Enables early stopping during training. | `true` |
| `early_stop.patience` | Number of epochs without F1-score improvement before stopping. | `15` |

### Data augmentation

| Parameter | Description | Reference |
| --- | --- | --- |
| `enabled` | Enables augmentation during training. | `true` |
| `prob` | Probability used by the augmentation selection logic. | `0.50` |
| `horizontal_flip` | Enables horizontal flipping. | `false` |
| `gaussian_noise` | Enables Gaussian-noise augmentation. | `false` |
| `random_erasing` | Enables random erasing. | `false` |
| `random_translation` | Enables random translation. | `true` |
| `max_translation` | Maximum translation magnitude in pixels. | `4` |
| `rotation` | Enables rotation. | `false` |
| `max_rotation` | Maximum rotation magnitude in degrees. | `5.0` |
| `border_erasing` | Enables border erasing. | `false` |
| `border_erasing_prob` | Probability associated with border erasing. | `0.5` |
| `gaussian_blur` | Enables Gaussian blur. | `false` |
| `blur_sigma` | Gaussian blur sigma. | `0.8` |
| `border_scale` | Enables border scaling. | `false` |
| `border_scale_px` | Border scaling amount in pixels. | `1` |

