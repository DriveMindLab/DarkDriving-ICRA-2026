import argparse
import os
import sys
from glob import glob
from typing import List

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix


class MergeCarValidator(DetectionValidator):
    merge_ids_src = [2, 5, 7]

    def init_metrics(self, model):
        super().init_metrics(model)
        self.names = {0: "car"}
        self.nc = 1
        self.metrics.names = self.names
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)

    def _prepare_batch(self, si, batch):
        prepared = super()._prepare_batch(si, batch)
        cls = prepared["cls"].reshape(-1).to(torch.int64)
        if cls.numel() and cls.max() == 0:
            keep = torch.ones_like(cls, dtype=torch.bool)
        else:
            keep = torch.isin(cls, torch.tensor(self.merge_ids_src, device=cls.device, dtype=torch.int64))
        prepared["cls"] = torch.zeros((int(keep.sum()),), device=prepared["cls"].device, dtype=torch.float32)
        prepared["bbox"] = prepared["bbox"][keep]
        return prepared

    def postprocess(self, preds):
        results = super().postprocess(preds)

        new_results = []
        for res in results:
            if torch.is_tensor(res):
                if res.ndim != 2 or res.shape[1] < 6:
                    cols = res.shape[1] if res.ndim == 2 else 6
                    new_results.append(torch.empty((0, cols), device=res.device, dtype=res.dtype))
                    continue

                merge_ids = torch.tensor(self.merge_ids_src, device=res.device, dtype=torch.int64)
                cls = res[:, 5].to(torch.int64)
                keep_mask = torch.isin(cls, merge_ids)

                if keep_mask.any():
                    filtered = res[keep_mask].clone()
                    filtered[:, 5] = 0.0
                    new_results.append(filtered)
                else:
                    new_results.append(res[:0])
                continue

            if not isinstance(res, dict):
                new_results.append(res)
                continue

            cls = res["cls"].to(torch.int64)
            merge_ids = torch.tensor(self.merge_ids_src, device=cls.device, dtype=torch.int64)
            keep_mask = torch.isin(cls, merge_ids)

            if not keep_mask.any():
                res["bboxes"] = torch.empty((0, 4), device=res["bboxes"].device)
                res["conf"] = torch.empty((0,), device=res["conf"].device)
                res["cls"] = torch.empty((0,), device=res["cls"].device)
                if "extra" in res:
                    res["extra"] = torch.empty((0, 0), device=res["bboxes"].device)
                new_results.append(res)
                continue

            res["bboxes"] = res["bboxes"][keep_mask]
            res["conf"] = res["conf"][keep_mask]
            res["cls"] = torch.zeros_like(res["conf"], dtype=torch.float32)
            if "extra" in res and res["extra"].ndim == 2 and res["extra"].shape[0] == cls.shape[0]:
                res["extra"] = res["extra"][keep_mask]

            new_results.append(res)
        return new_results


def abspath(path: str) -> str:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(path))

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_val_dir(data_yaml: str) -> (str, str):
    cfg = read_yaml(data_yaml)
    val = cfg.get("val", None)
    root = cfg.get("path", None)
    if val is None:
        raise ValueError(f"{data_yaml} does not define a 'val' path")

    if root:
        root = abspath(root)
        val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    else:
        val_dir = val if os.path.isabs(val) else abspath(val)

    val_dir = abspath(val_dir)
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"validation directory not found: {val_dir}")

    save_dir = root if root else os.path.dirname(val_dir) + '\predict'
    os.makedirs(save_dir, exist_ok=True)
    return val_dir, save_dir


def list_images(folder: str, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    files.sort()
    return files


def clear_dataset_cache_files(data_yaml: str) -> int:
    cfg = read_yaml(data_yaml)
    root = cfg.get("path")
    if root:
        root = abspath(root)
    else:
        root = os.path.dirname(data_yaml)

    patterns = [
        os.path.join(root, "**", "labels", "*.cache"),
        os.path.join(root, "**", "*.cache"),
    ]
    removed = 0
    seen = set()
    for pattern in patterns:
        for cache_path in glob(pattern, recursive=True):
            if cache_path in seen or not os.path.isfile(cache_path):
                continue
            seen.add(cache_path)
            try:
                os.remove(cache_path)
                removed += 1
                print(f"removed cache: {cache_path}")
            except OSError as ex:
                print(f"cache remove failed: {cache_path} ({ex})")
    return removed


def print_metrics(metrics, merge_car: bool = False):
    try:
        print(f"mAP50-95: {metrics.box.map:.4f}")
    except Exception:
        print("mAP50-95: unavailable")

    names = getattr(metrics, "names", None)
    maps = getattr(metrics.box, "maps", None) if hasattr(metrics, "box") else None

    if maps is None:
        return

    if merge_car:
        try:
            print(f"car AP50-95: {float(maps[0]):.4f}")
        except Exception:
            print("car AP50-95: unavailable")
        return

    for i, ap in enumerate(maps):
        cname = names[i] if isinstance(names, (list, tuple)) and i < len(names) \
            else (names.get(i, str(i)) if isinstance(names, dict) else str(i))
        try:
            print(f"{cname}: {float(ap):.4f}")
        except Exception:
            print(f"{cname}: {ap}")


def predict_and_save_flat(
    model: YOLO, src_dir: str, save_dir: str,
    imgsz: int = 640, conf: float = 0.1, iou: float = 0.50,
    merge_car: bool = False, limit: int = None
):

    os.makedirs(save_dir, exist_ok=True)

    gen = model.predict(
        source=src_dir,
        stream=True,
        batch=16,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=0,
        workers=0,
        verbose=False,
        save=False, save_txt=False, save_conf=False
    )

    processed = 0
    for result in tqdm(gen, desc="[Predict]"):
        if limit is not None and processed >= limit:
            break

        if merge_car:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                cls = boxes.cls.int().cpu().numpy()
                keep = np.isin(cls, [2, 5, 7])
                if keep.any():
                    boxes = boxes[keep]
                    boxes.cls[:] = 0
                    result.boxes = boxes
                else:
                    result.boxes = result.boxes[:0]

            result.names = {0: "car"}

        src_path = getattr(result, "path", "") or ""
        base = os.path.basename(src_path) or f"{processed:06d}.jpg"
        out_path = os.path.join(save_dir, base)

        rendered = result.plot()
        h, w = rendered.shape[:2]
        if max(h, w) != imgsz:
            scale = imgsz / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            rendered = cv2.resize(rendered, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(out_path, rendered)
        processed += 1

    print(f"saved {processed} predictions")


def main():
    parser = argparse.ArgumentParser(description="YOLO validation & flat prediction saver")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset .yaml (e.g., darkdriving_det.yaml)")
    parser.add_argument("--model_ckpt", type=str, default="yolo11x.pt", help="Path to YOLO checkpoint")
    parser.add_argument("--predict", action="store_true", help="Run prediction on val set and save rendered images flat into save_dir")
    parser.add_argument("--imgsz", type=int, default=512, help="Inference image size for prediction")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for prediction")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold for prediction")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    data_yaml = abspath(args.data)
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    data_cfg = read_yaml(data_yaml)
    merge_car = bool(data_cfg.get("merge_car", False))

    model_ckpt = abspath(args.model_ckpt)
    if not os.path.isfile(model_ckpt):
        print(f"weights not found: {model_ckpt}")
    print(f"model: {args.model_ckpt}")
    model = YOLO(args.model_ckpt)

    if args.train:
        print("train start")
        model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device="0",
        )
        print("train done")

    val_dir, save_dir = resolve_val_dir(data_yaml)
    print(f"val: {val_dir}")
    print(f"out: {save_dir}")

    if merge_car:
        print("val: car")
        try:
            metrics = model.val(data=data_yaml, validator=MergeCarValidator, verbose=False)
        except ModuleNotFoundError as ex:
            if "numpy._core" not in str(ex):
                raise
            print("clear cache and retry")
            clear_dataset_cache_files(data_yaml)
            metrics = model.val(data=data_yaml, validator=MergeCarValidator, verbose=False)

    else:
        print("val")
        try:
            metrics = model.val(data=data_yaml, save=True, cache=False, verbose=False)
        except ModuleNotFoundError as ex:
            if "numpy._core" not in str(ex):
                raise
            print("clear cache and retry")
            clear_dataset_cache_files(data_yaml)
            metrics = model.val(data=data_yaml, save=True, cache=False, verbose=False)

    print_metrics(metrics, merge_car=merge_car)

    if args.predict:
        predict_and_save_flat(
            model=model,
            src_dir=val_dir,
            save_dir=save_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            merge_car=merge_car
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
