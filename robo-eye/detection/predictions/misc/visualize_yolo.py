"""Visualizing the labels from the YOLO style files."""

import os
from pathlib import Path

import cv2


def visualize_labels(data_dir, labels_dir, class_names, label_type="GT"):
    """
    Visualize images with bounding boxes for ground truth or predicted labels.
    Args:
        data_dir (Path): Path to the directory containing 'images' subdirectory.
        labels_dir (Path): Path to the directory with ground truth or prediction labels.
        class_names (list): List of class names corresponding to class IDs.
        label_type (str): Type of labels to visualize ('GT' for ground truth, 'PRED' for predictions).
    """
    images_dir = data_dir / 'images'

    # Sort files in chronological order
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith(".png")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    for file_name in image_files:
        image_path = images_dir / file_name
        label_path = labels_dir / (os.path.splitext(file_name)[0] + ".txt")

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        img_height, img_width = image.shape[:2]

        labels = (
            load_prediction_labels(label_path)
            if label_type == "PRED"
            else load_ground_truth_labels(label_path)
        )

        draw_bounding_boxes(image, labels, class_names, img_width, img_height, label_type)

        cv2.imshow(f"Labeled Image - {label_type}", image)
        if cv2.waitKey(0) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()


def load_ground_truth_labels(label_file):
    """
    Load ground truth labels for a single image.
    Args:
        label_file (Path): Path to the ground truth label file.
    Returns:
        list: A list of bounding box dictionaries with 'class_id' and 'bbox'.
    """
    labels = []

    if not os.path.exists(label_file):
        return labels  # Return empty list if label file does not exist

    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])  # Class ID
            bbox = list(map(float, parts[1:5]))  # Normalized x_center, y_center, width, height
            labels.append({
                "class_id": class_id,
                "bbox": bbox
            })

    return labels


def load_prediction_labels(label_file):
    """
    Load YOLOv11 predictions for a single image.
    Args:
        label_file (Path): Path to the prediction label file.
    Returns:
        list: A list of bounding box dictionaries with 'class_id', 'prob', and 'bbox'.
    """
    labels = []

    if not os.path.exists(label_file):
        return labels  # Return empty list if label file does not exist

    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])  # Class ID
            prob = float(parts[1])  # Confidence probability
            bbox = list(map(float, parts[2:6]))  # Normalized x_center, y_center, width, height
            labels.append({
                "class_id": class_id,
                "prob": prob,
                "bbox": bbox
            })

    return labels


def draw_bounding_boxes(image, labels, class_names, img_width, img_height, label_type):
    """
    Draw bounding boxes and labels on an image.
    Args:
        image (ndarray): The image to annotate.
        labels (list): A list of label dictionaries with 'class_id' and 'bbox'.
        class_names (list): A list of class names.
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        label_type (str): Type of label (e.g., 'GT' for ground truth, 'PRED' for predictions).
    """
    for label in labels:
        class_id = label["class_id"]
        bbox = label["bbox"]

        # Bounding box values are in pixels directly
        x_center, y_center, width, height = bbox
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        if 0 <= class_id < len(class_names):
            class_label = class_names[class_id]
        else:
            class_label = f"Class {class_id}"

        # if label_type == "PRED" and "prob" in label:
        #     class_label += f" ({label['prob']:.2f})"

        color = (0, 255, 0)
        if class_id == 0:  # Pedestrian
            color = (255, 0, 0)
        elif class_id == 1:  # Cyclist
            color = (0, 0, 255)
        elif class_id == 2:  # Car
            color = (0, 255, 255)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw the class label
        cv2.putText(image, f"{class_label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent.parent / "datasets" / "seq_02" / "train"
    labels_dir = data_dir / 'labels'
    predicted_labels_dir = Path(__file__).parent / 'predictions' / 'labels_px'
    class_names = ["Pedestrian", "Cyclist", "Car"]

    # Set label_type to either 'GT' or 'PRED'
    label_type = "PRED"

    visualize_labels(data_dir, labels_dir if label_type == "GT" else predicted_labels_dir, class_names, label_type)
