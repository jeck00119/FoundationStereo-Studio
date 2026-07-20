"""Application entry point.  Run with:  python -m studio.app"""
from __future__ import annotations

import sys

# Must run before pyqtgraph / Qt are imported below: under pythonw.exe
# (run_studio.bat) sys.stdout/stderr are None and any library warning that
# writes to them would crash the GUI before it appears.
from ._streams import ensure_streams

ensure_streams("fs_studio_gui")

from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtWidgets import QApplication

# The 3D cloud tab is a QtWebEngine (Chromium) view. It shares an OpenGL context
# with the rest of Qt, and that attribute MUST be set before the QApplication is
# constructed — after is too late and the web view falls back to slow software GL.
QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

from .main_window import MainWindow
from .theme import apply_theme


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("FoundationStereo Studio")
    app.setOrganizationName("FSStudio")

    settings = QSettings("FSStudio", "FoundationStereoStudio")
    theme = settings.value("theme", "dark")
    if theme not in ("dark", "light"):
        theme = "dark"
    apply_theme(app, theme)

    win = MainWindow(theme)
    win.show()
    # ensure the engine child is stopped even on exit paths that skip closeEvent
    # (QApplication.quit(), an unhandled exception in a slot, etc.)
    app.aboutToQuit.connect(win.worker.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
