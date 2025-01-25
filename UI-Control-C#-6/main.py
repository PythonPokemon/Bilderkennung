from ultralytics import YOLO

# Modell trainniert auf overfitting Version 6 |
model = YOLO("C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\runs\\detect\\train\\weights\\best.pt")

# Starte das Training mit optimierten Parametern
# Version-6 OVerfitting
# C:\\Users\\Student\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-6\\config\\data.yaml

# Version-8 2.000 Bilder
# C:\\Users\\Student\\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-8\\data.yaml
model.train(
    data="C:\\Users\\Student\OneDrive - GFN GmbH (EDU)\\Desktop\\Jascha\\Programmieren\\Python Projekte\\Bilderkennung\\UI-Control-C#-6\\config\\data.yaml",
    epochs=100,                  # Mehr Epochen für bessere Konvergenz
    batch=16,                   # Mittelgroße Batch-Größe für bessere GPU-Auslastung
    lr0=0.005,                  # Niedrigere Start-Lernrate für stabileres Training
    optimizer="AdamW",          # Verbesserter Optimierer für robustes Training
    warmup_epochs=1,            # Warm-up für stabileren Start
    patience=100,                # Geduld für Early Stopping (10 Epochen ohne Verbesserung)
    save=True,                  # Speichere das Modell nach dem Training
    workers=8,                  # Maximale Parallelisierung bei der Datenvorbereitung
    verbose=True,            # Detaillierte Trainingsinformationen anzeigen
    augment=True,               # Datenaugmentation aktivieren
    scale=0.5,                  # Skaliere kleine Objekte stärker
    translate=0.1,              # Bildverschiebung zur Unterstützung teilweiser Sichtbarkeit
    fliplr=0.5,                 # Horizontale Spiegelung für mehr Variabilität
    auto_augment="randaugment", # Automatische Datenaugmentierung für verbesserte Generalisierung
    erasing=0.4,                # Löschen von Bildteilen zur Förderung von Robustheit
    crop_fraction=1.0,          # Volles Zuschneiden für fokussierte Objekterkennung
    exist_ok=True               # Überschreibe vorhandene Dateien ohne Fehler
)
