# Version-6 | 281 Bilder | Dauer pro Epoche 01:15 == mAP 0.99 | OVerfitting
# C:\\Users\\Student\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-6\\config\\data.yaml

# Version-8 | 758 Bilder | Dauer pro Epoche 04:30 == mAP 0.60 | 100 epochen
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-8\\config\\data.yaml

# Version-9 | 2375 Bilder | Dauer pro Epoche 15.05 == mAP 0.7 | 27 epochen
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-9\\config\\data.yaml

# Version-10 | 3074 Bilder | Dauer pro Epoche ??? == mAP ???
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-10\\config\\data.yaml

from ultralytics import YOLO

# Modell trainniert auf best.pt |
model = YOLO("C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\runs\\detect\\train\\weights\\best.pt")

# Starte das Training mit optimierten Parametern
model.train(
    data="C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-9\\config\\data.yaml",
    epochs=27,                  # Mehr Epochen für bessere Konvergenz
    batch=16,                   # Mittelgroße Batch-Größe für bessere GPU-Auslastung oder 64 | 0.7 == 70%
    #device=0,                  # ob man grafikkarte benutzen soll oder nicht
    lr0=0.005,                  # Niedrigere Start-Lernrate für stabileres Training |   0.01
    optimizer="auto",          # Verbesserter Optimierer für robustes Training
    warmup_epochs=1,            # Warm-up für stabileren Start
    patience=10,                # Geduld für Early Stopping (10 Epochen ohne Verbesserung)
    save=True,                  # Speichere das Modell nach dem Training
    workers=8,                  # Maximale Parallelisierung bei der Datenvorbereitung
    verbose=True,            # Detaillierte Trainingsinformationen anzeigen
    #augment=True,               # Datenaugmentation aktivieren
    scale=0.5,                  # Skaliere kleine Objekte stärker
    #translate=0.1,              # Bildverschiebung zur Unterstützung teilweiser Sichtbarkeit
    #fliplr=0.5,                 # Horizontale Spiegelung für mehr Variabilität
    #auto_augment="randaugment", # Automatische Datenaugmentierung für verbesserte Generalisierung
    #erasing=0.4,                # Löschen von Bildteilen zur Förderung von Robustheit
    crop_fraction=1.0,          # Volles Zuschneiden für fokussierte Objekterkennung
    exist_ok=True               # Überschreibe vorhandene Dateien ohne Fehler
)
