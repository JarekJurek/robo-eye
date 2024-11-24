from ultralytics import YOLO


def main():
    model = YOLO('yolo11n.pt')
    model.train(data="dataset.yaml", epochs=1, imgsz=640)
    model.export()


if __name__ == "__main__":
    main()
