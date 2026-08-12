"""Shared fixtures. The repo root is put on sys.path so `studio.*` imports
resolve when pytest is run from anywhere; the QApplication fixture serves the
few tests that touch real widgets (created once — Qt allows only one)."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QSettings must NOT be the user's real settings. Two ways that bites, both seen:
# a test constructing a MainWindow reads whatever ROI/boxes/model the user last
# left in the app — so the suite passes or fails depending on GUI state that has
# nothing to do with the code — and, worse, closeEvent WRITES, so running the
# tests could overwrite a carefully set up measurement config. Point the whole
# session at a throwaway directory before anything imports Qt.
_CFG = tempfile.mkdtemp(prefix="fs_test_settings_")
os.environ["XDG_CONFIG_HOME"] = _CFG
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def _isolated_settings():
    """Belt-and-braces: also force QSettings' own path, since Qt may resolve it
    from its own state rather than the environment."""
    from PySide6.QtCore import QSettings

    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, _CFG)
    QSettings.setDefaultFormat(QSettings.IniFormat)
    yield


@pytest.fixture(scope="session")
def qapp(_isolated_settings):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
