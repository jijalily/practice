import sys
from PyQt6.QtWidgets import QApplication
from qt_material import apply_stylesheet
from ui.main_window import MainWindow
from model.predictor import Predictor


app = QApplication(sys.argv)
apply_stylesheet(app, theme='dark_purple.xml') 

predictor = Predictor(
    "demo_app/weights/EfficientNet.pth"
)

window = MainWindow(predictor)
window.show()

sys.exit(app.exec())