import os
from pathlib import Path

import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ultralytics import YOLO


class MultiSeqDataset(Dataset):
    def __init__(self, sequences, transform=None):
        """
        Args:
            sequences (list): List of tuples where each tuple is (images_dir, labels_file).
            transform (callable, optional): Transform to be applied to the images.
        """
        self.sequences = sequences
        self.transform = transform
        self.data = self._load_all_labels()

    def _load_all_labels(self):
        data = []
        for images_dir, labels_file in self.sequences:
            labels = self._load_labels(labels_file)
            for frame_id in labels:
                data.append({
                    "image_path": os.path.join(images_dir, f"{frame_id:06d}.png"),
                    "annotations": labels[frame_id]
                })
        return data

    def _load_labels(self, labels_file):
        labels = {}
        with open(labels_file, "r") as file:
            for line in file:
                parts = line.strip().split()
                frame_id = int(parts[0])
                if frame_id not in labels:
                    labels[frame_id] = []
                labels[frame_id].append({
                    "track_id": int(parts[1]),
                    "type": parts[2],
                    "bbox": list(map(float, parts[6:10])),
                })
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = entry["image_path"]
        annotations = entry["annotations"]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, annotations


def create_dataloaders(base_dir, transform=None, batch_size=4):
    seq_01 = (os.path.join(base_dir, "seq_01/image_02/data"), os.path.join(base_dir, "seq_01/labels.txt"))
    seq_02 = (os.path.join(base_dir, "seq_02/image_02/data"), os.path.join(base_dir, "seq_02/labels.txt"))
    seq_03 = (os.path.join(base_dir, "seq_01/image_03/data"), os.path.join(base_dir, "seq_01/labels.txt"))

    train_sequences = [seq_01, seq_02]
    test_sequences = [seq_03]

    train_dataset = MultiSeqDataset(train_sequences, transform=transform)
    test_dataset = MultiSeqDataset(test_sequences, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_model(model, dataloader, epochs=10, imgsz=640, lr=0.001):
    model.train(
        data=dataloader,
        epochs=epochs,
        imgsz=imgsz,
        lr=lr,
    )
    return model


def main():
    project_dir = Path(__file__).parent
    base_data_dir = project_dir / "34759_final_project_rect"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 640)),
    ])

    train_loader, test_loader = create_dataloaders(base_data_dir, transform=transform)

    model = YOLO('yolo11n.pt')

    # Freeze the backbone for transfer learning
    for param in model.model.backbone.parameters():
        param.requires_grad = False

    model = train_model(model, train_loader, epochs=20)

    model.export(path="trained_yolov11.pt")

    # No testing since we don't have test ground-truth labels
    # for images, annotations in test_loader:
    #     detections = model(images)
    #     print("Detections:", detections)


if __name__ == "__main__":
    main()
