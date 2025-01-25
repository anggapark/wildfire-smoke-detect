import os
import argparse
from ultralytics import YOLO


def inference_model(input_dir, output_dir, weight, device):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(weight)

    results = model.predict(
        input_dir,
        save=True,
        project=output_dir,
        conf=0.5,
        iou=0.7,
        device=device,
    )

    output_path = results[0].save(filename=output_dir)

    return output_path


def main():
    # setup argument parsing
    parser = argparse.ArgumentParser(description="Wildfire smoke detection")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for detection results",
    )
    parser.add_argument(
        "--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--weights", type=str, default="yolo11n.pt", help="model weights path"
    )
    args = parser.parse_args()

    try:
        output_img = inference_model(
            input_dir=args.input,
            output_dir=args.output,
            weight=args.weights,
            device=args.device,
        )
        print(f"Detection complete. Output saved to: {output_img}")
    except Exception as e:
        print(f"Error during detection: {e}")


if __name__ == "__main__":
    main()
