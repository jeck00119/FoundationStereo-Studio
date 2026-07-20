"""Shared fixtures. The repo root is put on sys.path so `studio.*` imports
resolve when pytest is run from anywhere; the QApplication fixture serves the
few tests that touch real widgets (created once — Qt allows only one)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
