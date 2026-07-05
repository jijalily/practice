# report.py
# Модуль генерации краткого отчёта по результатам работы приложения.
# Реализует требование этапа 7 задания на практику:
# "генерацию краткого отчёта в PDF или Excel".

from pathlib import Path
from datetime import datetime

# Excel-экспорт (pandas + openpyxl)
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# PDF-экспорт (reportlab)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


class ReportGenerator:
    """Формирует отчёты по сохранённой истории запусков."""

    def __init__(self, records):
        self.records = records

    def _summary(self):
        """Сводная статистика по истории."""
        total = len(self.records)
        occupied = sum(
            1 for r in self.records
            if r.get("prediction") == "occupied"
        )
        free = total - occupied
        rate = (occupied / total * 100) if total else 0.0
        avg_time = (
            sum(r.get("inference_time_ms", 0) for r in self.records) / total
            if total else 0.0
        )
        avg_conf = (
            sum(r.get("confidence", 0) for r in self.records) / total
            if total else 0.0
        )
        return {
            "total": total,
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(rate, 2),
            "avg_inference_time_ms": round(avg_time, 2),
            "avg_confidence": round(avg_conf, 4)
        }

    def to_excel(self, output_path):
        """Сохраняет историю и сводку в книгу Excel (.xlsx)."""
        if not _HAS_PANDAS:
            raise RuntimeError(
                "Для экспорта в Excel установите pandas и openpyxl: "
                "pip install pandas openpyxl"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.records)

        if df.empty:
            df = pd.DataFrame(columns=[
                "timestamp", "image_name", "prediction",
                "confidence", "free_prob", "occupied_prob",
                "inference_time_ms"
            ])

        summary = self._summary()
        summary_df = pd.DataFrame([
            {"Показатель": "Всего запросов", "Значение": summary["total"]},
            {"Показатель": "Занято", "Значение": summary["occupied"]},
            {"Показатель": "Свободно", "Значение": summary["free"]},
            {"Показатель": "Загруженность, %", "Значение": summary["occupancy_rate"]},
            {"Показатель": "Среднее время инференса, мс", "Значение": summary["avg_inference_time_ms"]},
            {"Показатель": "Средняя уверенность", "Значение": summary["avg_confidence"]},
        ])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Сводка", index=False)
            df.to_excel(writer, sheet_name="История", index=False)

        return str(output_path)

    def to_pdf(self, output_path):
        """Сохраняет сводный отчёт в PDF."""
        if not _HAS_REPORTLAB:
            raise RuntimeError(
                "Для экспорта в PDF установите reportlab: "
                "pip install reportlab"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        elements = []

        elements.append(Paragraph(
            "Отчёт о работе системы определения занятости парковочных мест",
            styles["Title"]
        ))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(
            f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            styles["Normal"]
        ))
        elements.append(Spacer(1, 12))

        # Сводная таблица
        summary = self._summary()
        elements.append(Paragraph("Сводная статистика", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        summary_data = [
            ["Показатель", "Значение"],
            ["Всего запросов", str(summary["total"])],
            ["Занято", str(summary["occupied"])],
            ["Свободно", str(summary["free"])],
            ["Загруженность, %", str(summary["occupancy_rate"])],
            ["Среднее время инференса, мс", str(summary["avg_inference_time_ms"])],
            ["Средняя уверенность", str(summary["avg_confidence"])],
        ]

        summary_table = Table(summary_data, hAlign="LEFT")
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5c4b8a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 18))

        # Последние записи истории
        elements.append(Paragraph("Последние запуски", styles["Heading2"]))
        elements.append(Spacer(1, 6))

        recent = self.records[-15:]
        history_data = [["Время", "Изображение", "Прогноз", "Уверенность", "Время, мс"]]
        for r in recent:
            history_data.append([
                str(r.get("timestamp", "")),
                str(r.get("image_name", "")),
                str(r.get("prediction", "")),
                f"{r.get('confidence', 0) * 100:.1f}%",
                str(r.get("inference_time_ms", "")),
            ])

        history_table = Table(history_data, hAlign="LEFT", repeatRows=1)
        history_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5c4b8a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(history_table)

        doc.build(elements)
        return str(output_path)
