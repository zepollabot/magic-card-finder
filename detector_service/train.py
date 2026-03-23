"""Training script for the MTG card name detector.

Downloads a labelled dataset from Roboflow and fine-tunes a YOLO26 model.
Run inside the container:

    docker compose exec detector python train.py

Environment variables
---------------------
ROBOFLOW_API_KEY   : Required. Obtain at https://app.roboflow.com/settings/api
ROBOFLOW_WORKSPACE : Roboflow workspace slug  (default: proyectgraduacion)
ROBOFLOW_PROJECT   : Roboflow project slug    (default: mtg-cards-label)
ROBOFLOW_VERSION   : Dataset version number   (default: 5)
TRAIN_EPOCHS       : Training epochs           (default: 100)
TRAIN_BATCH        : Batch size                (default: 16)
TRAIN_DEVICE       : Device string             (default: cpu)
"""
import os
import sys

import yaml
from roboflow import Roboflow
from ultralytics import YOLO


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY environment variable is required.")
        sys.exit(1)

    workspace = os.getenv("ROBOFLOW_WORKSPACE", "pietros-workspace-lcffu")
    project_name = os.getenv("ROBOFLOW_PROJECT", "mtg-cards-custom")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "4"))
    epochs = int(os.getenv("TRAIN_EPOCHS", "100"))
    batch = int(os.getenv("TRAIN_BATCH", "16"))
    device = os.getenv("TRAIN_DEVICE", "cpu")
    workers = int(os.getenv("TRAIN_WORKERS", "0"))
    if device == "cpu" and workers > 0:
        print(f"WARNING: forcing workers=0 (was {workers}) — multiprocessing DataLoader on CPU adds overhead without GPU overlap benefit")
        workers = 0

    print(f"Downloading dataset: {workspace}/{project_name} v{version_num}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("yolov8")

    dataset_yaml = os.path.join(dataset.location, "data.yaml")
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = dataset.location
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"

    with open(dataset_yaml, "w") as f:
        yaml.dump(cfg, f)

    print(f"Classes: {cfg.get('names', {})}")
    print(f"Training: epochs={epochs}, batch={batch}, device={device}, workers={workers}")

    model = YOLO("yolo26n.pt")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        patience=20,
        device=device,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        workers=workers,
        project="/app/models",
        name="mtg_detector",
        exist_ok=True,
        save=True,
        plots=True,
    )

    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print("\nTraining complete!")
    print("  Best weights: /app/models/mtg_detector/weights/best.pt")
    if isinstance(map50, float):
        print(f"  mAP50: {map50:.3f}")
    else:
        print(f"  mAP50: {map50}")


if __name__ == "__main__":
    main()
