from ultralytics import YOLO


def main():
    model = YOLO('/zhome/a2/c/213547/robo-eye/runs/detect/train/weights/best.pt')
    model.train(data="dataset.yaml", epochs=100)
    model.export(dynamic=True)

if __name__ == "__main__":
    main()
