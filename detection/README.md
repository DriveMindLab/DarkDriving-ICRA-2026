# Detection for DarkDriving

This module provides a reproducible YOLO11-based detection pipeline for the
Dark Driving release. It includes validation scripts for the released and
fine-tuned checkpoints, dataset configuration files, and a lightweight command
line entry point for evaluation, fine-tuning, and rendered prediction export.

## Highlights

- YOLO11 detection evaluation with Ultralytics.
- Day, night, and SNR-aware low-light validation configs.
- Optional class remapping that merges COCO `car`, `bus`, and `truck` into a

## Repository Layout

```text
yolo11_det/
+-- config_release/       # 80-class validation configs with merge_car enabled
+-- config_release_ft/    # fine-tuned single-class car validation configs
+-- detection_dataset/    # expected dataset root
+-- yolo_det.py           # CLI for validation, training, and prediction export
+-- yolo11x.pt            # release checkpoint
+-- yolo11x_ft.pt         # fine-tuned checkpoint
+-- val_*.sh              # convenience evaluation scripts
```

The dataset YAML files use paths relative to their config directory. For
example:

```yaml
path: ../detection_dataset/night
train: images/train
val: images/val
```

Expected dataset layout:

```text
detection_dataset/
+-- day/
|   +-- images/{train,val}/
|   +-- labels/{train,val}/
+-- night/
|   +-- images/{train,val}/
|   +-- labels/{train,val}/
+-- snr-aware/night/
    +-- images/{train,val}/
    +-- labels/{train,val}/
```

## Installation

The project is pinned through `uv` and Python 3.9.

```sh
uv sync
```

If you are running on a different CUDA stack, update the PyTorch packages and
the `[[tool.uv.index]]` entry in `pyproject.toml` before syncing.

## Evaluation

Run all release-checkpoint validation splits:

```sh
uv run sh val_darkdriving_det_release.sh
```

Run all fine-tuned-checkpoint validation splits:

```sh
uv run sh val_darkdriving_det_release_ft.sh
```

Run a single split manually:

```sh
uv run python yolo_det.py \
  --data config_release/darkdriving_det_night.yaml \
  --model_ckpt yolo11x.pt
```

For `config_release/*`, `merge_car: true` enables car-centric evaluation by
mapping COCO classes `2`, `5`, and `7` to a single `car` class. The
`config_release_ft/*` configs are already single-class car configs.

## Fine-Tuning

Fine-tune from the release checkpoint:

```sh
uv run sh fine_tune.sh
```

Equivalent manual command:

```sh
uv run python yolo_det.py \
  --data config_release/darkdriving_det_night.yaml \
  --model_ckpt yolo11x.pt \
  --train \
  --epochs 100
```

Training uses the Ultralytics YOLO trainer with the image size provided by
`--imgsz` and GPU device `0`.

## Prediction Export

Add `--predict` to save rendered validation predictions into the dataset split
output directory:

```sh
uv run python yolo_det.py \
  --data config_release_ft/darkdriving_det_night.yaml \
  --model_ckpt yolo11x_ft.pt \
  --predict \
  --imgsz 512 \
  --conf 0.20 \
  --iou 0.45
```

The exporter writes images with the original basenames and keeps the output
flat, which is useful for side-by-side review or downstream report generation.

## Command Reference

```text
--data        Dataset YAML path. Required.
--model_ckpt  YOLO checkpoint path. Defaults to yolo11x.pt.
--train       Run training before validation.
--epochs      Number of training epochs. Defaults to 50.
--predict     Save rendered predictions from the validation image directory.
--imgsz       Image size for training and prediction. Defaults to 512.
--conf        Prediction confidence threshold. Defaults to 0.20.
--iou         Prediction NMS IoU threshold. Defaults to 0.45.
```

## Notes

- Keep checkpoint filenames aligned with the shell scripts, or pass an explicit
  `--model_ckpt` path.
- Ultralytics dataset cache files are cleared automatically and validation is
  retried if an incompatible cached NumPy module is detected.
