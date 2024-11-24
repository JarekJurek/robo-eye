from ultralytics import YOLO


def main():
    model = YOLO('yolo11n.pt')
    model.train(data="dataset.yaml", epochs=10)
    model.export(dynamic=True)


if __name__ == "__main__":
    main()
