import argparse
import os
from ultralytics import YOLO
from utils import plot_sample_image

def run_training(data_yaml, epochs, img_size, weights, output_dir="results/train", model_dir="models"):
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load model (use pre-trained weights or start from scratch)
    model = YOLO(weights)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project=output_dir,
        name="exp"
    )

    # Save best weights to models folder
    best_model_path = os.path.join(output_dir, "exp", "weights", "best.pt")
    if os.path.exists(best_model_path):
        final_path = os.path.join(model_dir, "best.pt")
        os.replace(best_model_path, final_path)
        print(f"✅ Training complete. Best model saved at: {final_path}")
    else:
        print("⚠️ Warning: best.pt not found after training.")

    # Optional: Show training results (metrics/loss curves)
    results_path = os.path.join(output_dir, "exp", "results.png")
    if os.path.exists(results_path):
        plot_sample_image(results_path, title="Training Metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Helmet Training Script")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Pretrained weights")

    args = parser.parse_args()

    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        img_size=args.img,
    )
