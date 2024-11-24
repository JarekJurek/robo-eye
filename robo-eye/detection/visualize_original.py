"""Visualizing the labels from the original labels.txt files."""
import os
from pathlib import Path

import cv2


def load_labels(labels_file, class_names):
    """Load labels from the label file and map class names to class IDs."""
    labels = {}
    class_map = {name: idx for idx, name in enumerate(class_names)}  # Map class names to IDs

    with open(labels_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            frame_id = int(parts[0])
            class_name = parts[2]  # Class name as string
            class_id = class_map[class_name]  # Map to class ID
            bbox = list(map(float, parts[6:10]))  # Normalized x_center, y_center, width, height

            if frame_id not in labels:
                labels[frame_id] = []
            labels[frame_id].append({
                "class_id": class_id,
                "bbox": bbox
            })
    return labels


def draw_bounding_boxes(image, labels, class_names):
    """Draw bounding boxes and labels on an image."""
    for label in labels:
        class_id = label["class_id"]
        bbox = label["bbox"]

        x1, y1, x2, y2 = map(int, bbox)  # Left top, right bottom

        color = (0, 255, 0)
        if class_id == 0:  # Car
            color = (0, 0, 255)
        elif class_id == 2:  # Cyclist
            color = (255, 0, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw the class label
        label = f"{class_names[class_id]}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def visualize_labels(image_dir, labels_file, class_names):
    """Visualize images with bounding boxes and class labels."""
    labels = load_labels(labels_file, class_names)

    for frame_id, label_list in labels.items():
        image_path = os.path.join(image_dir, f"{frame_id:06d}.png")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        draw_bounding_boxes(image, label_list, class_names)

        cv2.imshow("Labeled Image", image)
        if cv2.waitKey(0) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "seq_01" / "train"
    image_dir = data_dir / "images"
    labels_file = data_dir.parent / "labels.txt"
    class_names = ["Pedestrian", "Cyclist", "Car"]

    visualize_labels(image_dir, labels_file, class_names)
