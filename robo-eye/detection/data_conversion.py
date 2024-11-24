import os
from pathlib import Path

import cv2

def convert_kitti_to_yolo11(input_labels, output_dir, images_dir, class_mapping):
    """Converts KITTI-style labels to YOLOv11 format."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    with open(input_labels, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        frame_id = int(parts[0])  # Frame ID
        object_type = parts[2]  # Object type
        left, top, right, bottom = map(float, parts[6:10])  # Bounding box in absolute pixel coordinates

        # Skip objects with types not in class_mapping
        if object_type not in class_mapping:
            continue

        class_id = class_mapping[object_type]

        img_path = os.path.join(images_dir, f"{frame_id:010d}.png")
        if not os.path.exists(img_path):
            print(f"Image {img_path} not found. Skipping.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_path}. Skipping.")
            continue

        img_height, img_width = img.shape[:2]

        # Normalize bounding box coordinates
        x_center = (left + right) / 2 / img_width
        y_center = (top + bottom) / 2 / img_height
        width = (right - left) / img_width
        height = (bottom - top) / img_height

        # YOLOv11 format: <class_id> <x_center> <y_center> <width> <height>
        yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

        label_file = os.path.join(output_dir, f"{frame_id:010d}.txt")
        with open(label_file, 'a') as label_out:
            label_out.write(yolo_label)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent/ "datasets" / "seq_02"
    input_labels = data_dir / "labels.txt"
    images_dir = data_dir / "data"

    output_dir = data_dir / "labels"

    class_mapping = {
        "Pedestrian": 0,
        "Cyclist": 1,
        "Car": 2
    }

    convert_kitti_to_yolo11(input_labels, output_dir, images_dir, class_mapping)
