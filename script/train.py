import argparse
from pathlib import Path
from ultralytics import YOLO
from roboflow import Roboflow

# from .data.dataset import get_dataset
from clearml import Task
import warnings

warnings.filterwarnings("ignore")


def train_model(data, device, batch, epochs, img_size=640):
    # dataset = get_dataset(version=1)

    # Define parameters
    # dataset_yaml = f"../data/data.yaml"
    project = f"model"
    name = "smoke-detect"
    lr = 1e-3
    optimizer = "AdamW"

    # # create ClearML Task
    # hyp_task = Task.init(project_name=project, task_name=name)

    model_variant = "yolo11n.pt"
    # hyp_task.set_parameter("model_variant", model_variant)
    model = YOLO(model_variant, task="detect")

    # put all argument in the dict and pass it to ClearML
    args = dict(
        data=data,
        epochs=epochs,
        batch=batch,
        device=device,
        project=project,
        name=name,
        lr0=lr,
        imgsz=img_size,
        optimizer=optimizer,
        cache=True,
    )
    # hyp_task.connect(args)

    model.train(**args)

    # hyp_task.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 model to detect wildfire smoke"
    )
    parser.add_argument("data", type=str, default="data.yml", help="data for training")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch size for training")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--img_size", type=int, default=640, help="Image size for training"
    )

    # parse arguments
    args = parser.parse_args()

    # call train_model function with parsed arguments
    train_model(
        data=args.data,
        device=args.device,
        batch=args.batch,
        epochs=args.epochs,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
