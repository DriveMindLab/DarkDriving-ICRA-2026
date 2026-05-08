# DarkDriving LLE Runner

This project provides training and testing scripts for `dataset_darkdriving_lle`.
It uses the SNR-aware low-light enhancement model and expects paired day/night images.

## Environment

Create the virtual environment and install dependencies with `uv`:

```bash
uv sync --python 3.8
```

Equivalent script:

```bash
scripts/setup_env.sh
```

## Dataset Layout

Set `dataroot` in `options/train/darkdriving_lle.yml` and `options/test/darkdriving_lle.yml` to folders with this layout:

```text
darkdriving_lle/
  train/
    day/
      *.jpg
    night/
      *.jpg
  test/
    day/
      *.jpg
    night/
      *.jpg
```

The loader pairs images by sorted filename. `night/` is used as input and `day/` is used as ground truth.

## Configuration

Training config:

```text
options/train/darkdriving_lle.yml
```

Important fields:

```yaml
datasets:
  train:
    dataroot: ../data/darkdriving_lle/train/
  val:
    dataroot: ../data/darkdriving_lle/test/
```

Testing config:

```text
options/test/darkdriving_lle.yml
```

Important fields:

```yaml
datasets:
  test:
    dataroot: ../data/darkdriving_lle/test/
path:
  pretrain_model_G: ./experiments/darkdriving_release/models/latest_G.pth
```

## Scripts


Train with the default config:

```bash
scripts/train_darkdriving.sh
```

Train with another config:

```bash
OPT=options/train/darkdriving_lle.yml scripts/train_darkdriving.sh
```

Test with the default config:

```bash
scripts/test_darkdriving.sh
```

Test and choose an output folder:

```bash
SAVE_DIR=results/darkdriving_release scripts/test_darkdriving.sh
```

Test with another config:

```bash
OPT=options/test/darkdriving_lle.yml SAVE_DIR=results/my_run scripts/test_darkdriving.sh
```

The scripts use `.venv/bin/python` by default. You can override the Python executable:

```bash
PYTHON=/path/to/python scripts/train_darkdriving.sh
```

## Outputs

By default, training keeps the output simple:

```text
experiments/darkdriving_release/
  train_darkdriving_release.log
  models/
    latest_G.pth
```

Intermediate checkpoints are disabled by default. To save periodic checkpoints, set `logger.save_checkpoint_freq` in `options/train/darkdriving_lle.yml` to a positive iteration number, for example `5000`.

Training state files are disabled by default. To save resumable optimizer/scheduler states, set:

```yaml
logger:
  save_training_state: true
```

Test output images and metric text files are written under `SAVE_DIR`.

## Acknowledgement

This project is based on SNR-Aware Low-Light Enhancement.
