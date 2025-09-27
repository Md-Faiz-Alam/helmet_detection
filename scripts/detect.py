import argparse
import os
from datetime import datetime
from ultralytics import YOLO
from utils import create_light_bundle

def run_detection(weights, source, conf, n_samples, output_dir="results/detect"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Unique folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(output_dir, f"run_{timestamp}")

    # Load model
    model = YOLO(weights)

    # Run detection
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        project=run_folder,
        name="predictions",
        max_det=n_samples if n_samples > 0 else None
    )

    print(f"✅ Detection complete. Results saved in: {run_folder}")

    # Create light bundle zip
    bundle_path = create_light_bundle(run_folder)
    print(f"✅ Light bundle created at: {bundle_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Helmet Detection Script")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image/video folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples to process (0 = all)")

    args = parser.parse_args()

    run_detection(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        n_samples=args.n_samples
    )
