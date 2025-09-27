import os
import zipfile
import cv2
import matplotlib.pyplot as plt

def create_light_bundle(run_folder, bundle_dir="light_bundles"):
    """Create a light bundle (zip) of prediction images."""
    os.makedirs(bundle_dir, exist_ok=True)
    bundle_name = f"bundle_{os.path.basename(run_folder)}.zip"
    bundle_path = os.path.join(bundle_dir, bundle_name)

    with zipfile.ZipFile(bundle_path, "w") as zipf:
        for root, _, files in os.walk(run_folder):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    zipf.write(os.path.join(root, file), arcname=file)

    return bundle_path


def plot_sample_image(image_path, title="Sample Detection Overlay"):
    """Plot a sample image using matplotlib."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()
