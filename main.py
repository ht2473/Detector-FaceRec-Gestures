"""Application entry point."""
import sys
import traceback

from loguru import logger
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QFont

from src.gui.window import MainWindow


def _global_exception_handler(exctype, value, tb):
    """Catch unhandled exceptions, log them and show a dialog before exiting."""
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    logger.critical(f"Uncaught exception:\n{error_msg}")
    box = QMessageBox()
    box.setIcon(QMessageBox.Icon.Critical)
    box.setWindowTitle("Fatal Error")
    box.setText("An unexpected error occurred. The application will now close.")
    box.setDetailedText(error_msg)
    box.exec()
    sys.exit(1)


def main() -> None:
    sys.excepthook = _global_exception_handler

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
