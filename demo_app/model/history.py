# history.py
# Модуль сохранения истории запусков приложения в JSON.
# Реализует требование этапа 7 задания на практику:
# "сохранение истории запусков или результатов в JSON/БД".

import json
from pathlib import Path
from datetime import datetime


class HistoryManager:
    """Сохраняет результаты каждого предсказания в JSON-файл.

    Файл имеет структуру:
        {
            "runs": [
                {
                    "timestamp": "2026-07-03T19:00:00",
                    "image_name": "GOPR0025.JPG",
                    "prediction": "occupied",
                    "confidence": 0.9876,
                    "free_prob": 0.0124,
                    "occupied_prob": 0.9876,
                    "inference_time_ms": 45.2
                },
                ...
            ]
        }
    """

    def __init__(self, history_path):
        self.path = Path(history_path)
        self.records = self._load()

    def _load(self):
        """Загружает историю из файла, если он существует."""
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("runs", [])
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save(self):
        """Записывает текущую историю на диск."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"runs": self.records},
                f,
                ensure_ascii=False,
                indent=2
            )

    def add(
        self,
        image_name,
        prediction,
        confidence,
        free_prob,
        occupied_prob,
        inference_time_ms
    ):
        """Добавляет запись о предсказании и сохраняет её на диск."""
        label = "occupied" if prediction == 1 else "free"

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_name": image_name,
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "free_prob": round(float(free_prob), 4),
            "occupied_prob": round(float(occupied_prob), 4),
            "inference_time_ms": round(float(inference_time_ms), 2)
        }

        self.records.append(record)
        self._save()

        return record

    def clear(self):
        """Полностью очищает историю."""
        self.records = []
        self._save()

    def stats(self):
        """Возвращает сводную статистику по всей истории."""
        total = len(self.records)
        occupied = sum(
            1 for r in self.records
            if r["prediction"] == "occupied"
        )
        free = total - occupied
        rate = (occupied / total * 100) if total else 0.0

        avg_time = 0.0
        if total:
            avg_time = sum(
                r["inference_time_ms"] for r in self.records
            ) / total

        return {
            "total": total,
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(rate, 2),
            "avg_inference_time_ms": round(avg_time, 2)
        }
