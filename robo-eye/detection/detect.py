import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def save_labels(results, labels_dir, image_name):
    """
    Save YOLOv11 detection results to a text file in YOLO format.
    Args:
        results (Results): YOLOv11 prediction results.
        labels_dir (Path): Path to the directory for saving label files.
        image_name (str): Name of the image file (used for naming the label file).
    """

    labels_dir.mkdir(parents=True, exist_ok=True)

    label_file_path = labels_dir / f"{Path(image_name).stem}.txt"
    with open(label_file_path, 'w') as f:
        for box in results.boxes:
            cls = int(box.cls[0])  # Class ID
            conf = float(box.conf[0])  # Confidence score
            xywhn = box.xywhn[0].tolist()  # Normalized bbox as [x_center, y_center, width, height]

            x_center, y_center, width, height = xywhn
            f.write(f"{cls} {conf:.4f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def visualize_results(results, vis_dir, image_name):
    """
    Visualize YOLOv11 detection results and save the annotated image.
    Args:
        results (Results): YOLOv11 prediction results.
        vis_dir (Path): Path to the directory for saving visualized images.
        image_name (str): Name of the image file (used for naming the visualized image).
    """
    vis_dir.mkdir(parents=True, exist_ok=True)

    annotated_img = results.plot()

    vis_image_path = vis_dir / image_name
    cv2.imwrite(str(vis_image_path), annotated_img)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "combo" / "test"
    image_dir = data_dir / 'images'
    output_dir = Path(__file__).parent / 'predictions'
    vis_dir = Path(output_dir) / 'visualizations'
    labels_dir = Path(output_dir) / 'labels'

    if not Path(image_dir).exists():
        print(f"Image directory {image_dir} does not exist.")
        exit()

    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)

    # model = YOLO('yolo11n.pt')
    model = YOLO('/zhome/a2/c/213547/robo-eye/runs/detect/train2/weights/best.pt')

    for image_path in Path(image_dir).glob('*.*'):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        results = model.predict(source=img, conf=0.25)

        save_labels(results[0], labels_dir, image_path.name)
        visualize_results(results[0], vis_dir, image_path.name)
