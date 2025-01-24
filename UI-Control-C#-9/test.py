from ultralytics import YOLO

# Erstellen eines YOLO-Modells
model = YOLO("C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\runs\\detect\\train\\weights\\best.pt")  # Das Standardmodell, das verwendet wird (z.B. yolov8n.pt)

# Training des Modells
model.train(data="C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-6\\config\\data.yaml",
epochs=350,               # Epochenanzahl für längeres Training
batch=16,                # Mittelgroße Batch-Größe für Stabilität und Speicherverwaltung
lr0=0.005,               # Angepasste Start-Lernrate für stabilere Konvergenz
optimizer="AdamW",       # AdamW-Optimierer für robustes Lernen
warmup_epochs=1,         # Warm-up-Phase zur Optimierung des Starttrainings
patience=100,            # Geduld für Lernstop bei Stagnation
save=True,               # Automatisches Speichern des Modells nach Training
workers=8,               # Maximale Datenparallelisierung
augment=True,            # Aktivierung der Datenaugmentation
scale=0.5,               # Verstärkte Skalierung kleiner Objekte
translate=0.1,           # Bildverschiebung für teilverdeckte Objekte
fliplr=0.5,              # Horizontale Spiegelung für erhöhte Variabilität
auto_augment="randaugment",  # Automatische Augmentierung für Generalisierung
erasing=0.4,             # Teile des Bildes entfernen, um Robustheit zu fördern
crop_fraction=1.0,       # Vollständiges Zuschneiden für fokussierte Erkennung
exist_ok=True            # Überschreibung bestehender Ergebnisse zulassen
)