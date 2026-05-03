import sys
import traceback
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont
from src.gui.window import MainWindow
from loguru import logger

def global_exception_handler(exctype, value, tb):
    """Global exception handler to catch Qt crashes silently"""
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"Uncaught exception: {error_msg}")
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Icon.Critical)
    msg_box.setWindowTitle("Fatal Error")
    msg_box.setText("An unexpected error occurred!")
    msg_box.setDetailedText(error_msg)
    msg_box.exec()
    sys.exit(1)

def main():
    sys.excepthook = global_exception_handler
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()