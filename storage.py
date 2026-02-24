import csv
from datetime import datetime
from pathlib import Path

FILE = "history.csv"


def save_entry(text, level, stress_percent, lang):
    file_exists = Path(FILE).exists()

    with open(FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # écrire header si fichier n'existe pas
        if not file_exists:
            writer.writerow([
                "text",
                "stress_level",
                "stress_percent",
                "language",
                "time"
            ])

        writer.writerow([
            text,
            level,
            round(stress_percent, 2),
            lang,
            datetime.now().isoformat()
        ])


