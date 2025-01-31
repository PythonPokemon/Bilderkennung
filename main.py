# Version-6 | 281 Bilder | Dauer pro Epoche 01:15 == mAP 0.99 | OVerfitting?
# C:\\Users\\Student\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-6\\config\\data.yaml

# Version-8 | 758 Bilder | Dauer pro Epoche 04:30 == mAP 0.60 | 100 epochen
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-8\\config\\data.yaml

# Version-9 | 2375 Bilder | Dauer pro Epoche 15.05 == mAP 0.7 | 27 epochen
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-9\\config\\data.yaml

# Version-10 | 3074 Bilder | Dauer pro Epoche 25min == mAP 0.44 | 18 epochen , abbruch weil keine beserung!
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-10\\config\\data.yaml

from ultralytics import YOLO

# Modell trainniert auf best.pt |
model = YOLO("C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\runs\\detect\\train\\weights\\best.pt")

# Starte das Training mit optimierten Parametern
model.train(
    data="C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-1\\config\\data.yaml",
    epochs=150,                  # Mehr Epochen für bessere Konvergenz
    batch=16,                    # Batch-Größe anpassen (abhängig von deiner GPU)
    imgsz=640,                   # Bildgröße beibehalten
    lr0=0.01,                    # Start-Lernrate
    lrf=0.1,                     # Finale Lernrate (10% von lr0)
    momentum=0.937,              # Momentum für den Optimierer
    weight_decay=0.0005,         # Gewichtsverlust zur Regularisierung
    warmup_epochs=15,             # Warm-up-Epochen für stabileren Start
    patience=20,                 # Early Stopping nach 20 Epochen ohne Verbesserung
    augment=True,                # Datenaugmentierung aktivieren
    hsv_h=0.015,                 # Farbton-Variation (HSV-Augmentierung)
    hsv_s=0.7,                   # Sättigungs-Variation
    hsv_v=0.4,                   # Helligkeits-Variation
    translate=0.1,               # Bildverschiebung
    scale=0.5,                   # Skalierung
    fliplr=0.5,                  # Horizontale Spiegelung
    mosaic=1.0,                  # Mosaic-Augmentierung (100% Wahrscheinlichkeit)
    mixup=0.1,                   # Mixup-Augmentierung (10% Wahrscheinlichkeit)
    copy_paste=0.1,              # Copy-Paste-Augmentierung (10% Wahrscheinlichkeit)
    erasing=0.4,                 # Random Erasing (40% Wahrscheinlichkeit)
    workers=8,                   # Anzahl der Worker für Datenladen
    #device=0,                    # GPU verwenden (falls verfügbar)
    save=True,                   # Modell speichern
    exist_ok=True,               # Vorhandene Dateien überschreiben
    verbose=True                 # Detaillierte Ausgaben während des Trainings
)
