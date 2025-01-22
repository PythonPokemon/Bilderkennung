from ultralytics import YOLO

# Erstellen eines YOLO-Modells
model = YOLO("yolov8n.pt")  # Das Standardmodell, das verwendet wird (z.B. yolov8n.pt)

# Training des Modells
model.train(data="C:\\Users\\Student\OneDrive - GFN GmbH (EDU)\Desktop\\Jascha\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-9\\config\\data.yaml",
            epochs=50, batch=16, imgsz=640)  # Passen Sie die Parameter nach Bedarf an
