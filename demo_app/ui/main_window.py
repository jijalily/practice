from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QProgressBar,
    QGridLayout
)
from model.gradcam import GradCAMGenerator


class MainWindow(QWidget):

    def __init__(self, predictor):
        super().__init__()

        self.predictor = predictor
        self.image_path = None

        self.total_requests = 0
        self.occupied_count = 0
        self.free_count = 0

        self.gradcam = GradCAMGenerator(
            model=self.predictor.model,
            target_layer=self.predictor.model.features[-1],
            device=self.predictor.device
        )

        self.setWindowTitle(
            "Система определения занятости парковочных мест"
        )

        self.init_ui()

    def init_ui(self):

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(15)
        title = QLabel("Система определения занятости парковочных мест")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        root.addWidget(title)

        content = QHBoxLayout()
        content.setSpacing(15)
        root.addLayout(content)

        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_layout = QVBoxLayout(left_panel)
        image_title = QLabel("Исходное изображение")
        left_layout.addWidget(image_title)

        self.image_label = QLabel()
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.image_label.setMinimumSize(700, 500)
        self.image_label.setText(
            "Загрузите изображение"
        )
        left_layout.addWidget(self.image_label)
        content.addWidget(left_panel, 3)

        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_layout = QVBoxLayout(right_panel)

        result_title = QLabel("Результат анализа")
        right_layout.addWidget(result_title)

        self.status_label = QLabel("Нет данных")
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.status_label.setMinimumHeight(70)
        right_layout.addWidget(self.status_label)

        confidence_title = QLabel(
            "Уверенность модели"
        )
        right_layout.addWidget(confidence_title)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFormat(
            "%p%"
        )
        right_layout.addWidget(
            self.confidence_bar
        )

        stats_frame = QFrame()
        stats_layout = QGridLayout(stats_frame)
        stats_layout.addWidget(QLabel("Свободно:"),0,0)

        self.free_label = QLabel("-")
        stats_layout.addWidget(self.free_label,0,1)

        stats_layout.addWidget(QLabel("Занято:"),1,0)
        self.busy_label = QLabel("-")
        stats_layout.addWidget(self.busy_label,1,1)

        stats_layout.addWidget(QLabel("Время инференса:"),2,0)
        self.time_label = QLabel("-")
        stats_layout.addWidget(self.time_label,2,1)

        stats_layout.addWidget(QLabel("Архитектура:"),3,0)
        self.model_label = QLabel("EfficientNetV2-M")
        stats_layout.addWidget(self.model_label,3,1)

        right_layout.addWidget(stats_frame)

        self.global_stats_frame = QFrame()
        self.global_stats_frame.setObjectName("card")

        global_layout = QVBoxLayout(self.global_stats_frame)
        title = QLabel("Общая статистика")
        global_layout.addWidget(title)

        grid = QGridLayout()

        self.total_label = QLabel("0")
        self.occupied_total_label = QLabel("0")
        self.free_total_label = QLabel("0")
        self.occupancy_rate_label = QLabel("0%")

        grid.addWidget(QLabel("Всего запросов:"), 0, 0)
        grid.addWidget(self.total_label, 0, 1)

        grid.addWidget(QLabel("Занято (всего):"), 1, 0)
        grid.addWidget(self.occupied_total_label, 1, 1)

        grid.addWidget(QLabel("Свободно (всего):"), 2, 0)
        grid.addWidget(self.free_total_label, 2, 1)

        grid.addWidget(QLabel("Загруженность:"), 3, 0)
        grid.addWidget(self.occupancy_rate_label, 3, 1)

        global_layout.addLayout(grid)

        right_layout.addWidget(self.global_stats_frame)
        right_layout.addStretch()
        content.addWidget(right_panel, 1)

        gradcam_card = QFrame()
        gradcam_card.setObjectName("card")
        gradcam_layout = QVBoxLayout(gradcam_card)
        gradcam_title = QLabel("Визуализация внимания модели (Grad-CAM)")
        gradcam_layout.addWidget(gradcam_title)

        self.gradcam_label = QLabel()
        self.gradcam_label.setMinimumHeight(250)

        self.gradcam_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.gradcam_label.setText("Grad-CAM появится после анализа")
        gradcam_layout.addWidget(self.gradcam_label)
        root.addWidget(gradcam_card)

        buttons = QHBoxLayout()
        self.open_btn = QPushButton("Выбрать изображение")
        self.predict_btn = QPushButton("Анализировать")
        self.predict_btn.setEnabled(False)

        buttons.addWidget(self.open_btn)
        buttons.addWidget(self.predict_btn)

        root.addLayout(buttons)

        self.open_btn.clicked.connect(
            self.load_image
        )

        self.predict_btn.clicked.connect(
            self.predict_image
        )

    def load_image(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор изображения",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not path:
            return

        self.image_path = path
        pixmap = QPixmap(path)

        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        self.predict_btn.setEnabled(True)
    
    def predict_image(self):

        if not self.image_path:
            return
        
        self.total_requests += 1

        result = self.predictor.predict(self.image_path)

        image = result["image"]
        tensor = result["tensor"]

        cam = self.gradcam.generate(
            image_tensor=tensor,
            original_image=image,
            class_idx=result["prediction"]
        )

        bytes_per_line = 3 * cam.shape[1]

        qt_image = QImage(
            cam.data,
            cam.shape[1],
            cam.shape[0],
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        self.gradcam_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.gradcam_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
        )

        occupied = (result["prediction"] == 1)
        if occupied:
            self.occupied_count += 1
        else:
            self.free_count += 1

        if occupied:
            self.status_label.setText("ЗАНЯТО")

        else:
            self.status_label.setText("СВОБОДНО")

        self.confidence_bar.setValue(
            int(result["confidence"] * 100)
        )

        self.free_label.setText(
            f"{result['free_prob'] * 100:.2f}%"
        )
        self.busy_label.setText(
            f"{result['occupied_prob'] * 100:.2f}%"
        )
        self.time_label.setText(
            f"{result['time_ms']} мс"
        )

        self.total_label.setText(str(self.total_requests))
        self.occupied_total_label.setText(str(self.occupied_count))
        self.free_total_label.setText(str(self.free_count))
        if self.total_requests > 0:
            rate = (self.occupied_count / self.total_requests) * 100
        else:
            rate = 0

        self.occupancy_rate_label.setText(f"{rate:.2f}%")