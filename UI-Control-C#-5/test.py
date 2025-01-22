from roboflow import Roboflow
from ultralytics import YOLO

# Verbindung zu Roboflow herstellen und Dataset herunterladen
rf = Roboflow(api_key="i69F4gJn4h97e9d0s4XC")
project = rf.workspace("bildererkennung-im-bild").project("ui-control-c")
version = project.version(5)
dataset = version.download("yolov11")

# Pfad zum heruntergeladenen Dataset
dataset_path = r"C:\Users\Student\OneDrive - GFN GmbH (EDU)\Desktop\Jascha\Programmieren\Python Projekte\Bilderkennung\UI-Control-C#-5"



# Erstellen eines YOLO-Modells
model = YOLO("yolov8n.pt")  # Das Standardmodell, das verwendet wird (z.B. yolov8n.pt)

# Training des Modells
model.train(data=dataset_path, epochs=50, batch=16, imgsz=640)  # Passen Sie die Parameter nach Bedarf an
