"""Dark / light theming — a considered palette (cool slate + warm coral accent)
applied via both QPalette (so native bits look right) and a QSS stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

MONO = '"Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace'
SANS = '"Segoe UI", "Inter", system-ui, sans-serif'

DARK = {
    "bg": "#0C0F16", "surface": "#141926", "surface2": "#1B2130", "surface3": "#242C3D",
    "border": "#2A3242", "border_soft": "#1E2533",
    "text": "#E7ECF4", "muted": "#8A93A7", "faint": "#5A6379",
    "accent": "#FF6A3D", "accent_press": "#E85427", "accent_soft": "#2A2030",
    "accent2": "#2AA9E0", "ok": "#46C08A", "warn": "#E8B54A", "crit": "#F0596B",
    "sel": "#33251E",
}
LIGHT = {
    "bg": "#EEF1F7", "surface": "#FFFFFF", "surface2": "#F3F5FA", "surface3": "#E4E9F2",
    "border": "#D3DAE6", "border_soft": "#E4E9F2",
    "text": "#161C29", "muted": "#5A6577", "faint": "#93A0B4",
    "accent": "#E9542A", "accent_press": "#C8431D", "accent_soft": "#FBE9E2",
    "accent2": "#1687C0", "ok": "#2E9E72", "warn": "#C9932E", "crit": "#D64459",
    "sel": "#FBE3D9",
}

PALETTES = {"dark": DARK, "light": LIGHT}


def _qpalette(p: dict) -> QPalette:
    q = QPalette()
    q.setColor(QPalette.Window, QColor(p["bg"]))
    q.setColor(QPalette.WindowText, QColor(p["text"]))
    q.setColor(QPalette.Base, QColor(p["surface2"]))
    q.setColor(QPalette.AlternateBase, QColor(p["surface"]))
    q.setColor(QPalette.Text, QColor(p["text"]))
    q.setColor(QPalette.Button, QColor(p["surface2"]))
    q.setColor(QPalette.ButtonText, QColor(p["text"]))
    q.setColor(QPalette.ToolTipBase, QColor(p["surface3"]))
    q.setColor(QPalette.ToolTipText, QColor(p["text"]))
    q.setColor(QPalette.PlaceholderText, QColor(p["faint"]))
    q.setColor(QPalette.Highlight, QColor(p["accent"]))
    q.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    q.setColor(QPalette.Link, QColor(p["accent2"]))
    dis = QColor(p["faint"])
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        q.setColor(QPalette.Disabled, role, dis)
    return q


def _qss(p: dict) -> str:
    return f"""
    * {{ font-family: {SANS}; font-size: 13px; color: {p['text']}; }}
    *:disabled {{ color: {p['faint']}; }}
    QMainWindow, QWidget#Pane, QDialog {{ background: {p['bg']}; }}
    QWidget#LeftPane, QWidget#RightPane {{ background: {p['surface']}; }}
    QWidget#LeftPane {{ border-right: 1px solid {p['border']}; }}
    QWidget#RightPane {{ border-left: 1px solid {p['border']}; }}

    QLabel#Brand {{ font-family: {MONO}; font-weight: 600; font-size: 14px; }}
    QLabel[role="section"] {{ font-family: {MONO}; font-size: 10px; font-weight: 600;
        letter-spacing: 2px; color: {p['faint']}; }}
    /* foldable category header (CollapsibleSection) */
    QLabel#SectionArrow {{ color: {p['muted']}; font-size: 11px; }}
    QWidget#SectionHead {{ border-radius: 6px; }}
    QWidget#SectionHead:hover {{ background: {p['surface2']}; }}
    QLabel[role="muted"] {{ color: {p['muted']}; }}
    QLabel[role="mono"] {{ font-family: {MONO}; color: {p['text']}; }}
    QLabel[role="value"] {{ font-family: {MONO}; font-weight: 600; color: {p['text']}; }}
    /* the winning figure in a Compare column (only where 'better' is unambiguous) */
    QLabel[role="value"][stat="best"] {{ color: {p['ok']}; font-weight: 700; }}
    QLabel[role="value"][stat="warn"] {{ color: {p['warn']}; }}
    /* metric card (Analyze readouts): faint eyebrow + big headline value + a tidy
       key/value table, so a measurement reads as a card not a wall of mono text */
    QFrame#InfoCard {{ background: {p['surface2']}; border: 1px solid {p['border_soft']};
        border-radius: 9px; }}
    QLabel#CardHeadline {{ font-family: {MONO}; font-size: 20px; font-weight: 700;
        color: {p['accent']}; }}
    QLabel#CardUnit {{ font-family: {MONO}; font-size: 11px; font-weight: 600; color: {p['muted']}; }}
    QFrame#InfoRule {{ background: {p['border_soft']}; border: none; }}
    QLabel#Pill {{ font-family: {MONO}; font-size: 9px; font-weight: 700; letter-spacing: 1px;
        padding: 1px 7px; border-radius: 8px; }}
    QLabel#Pill[pill="live"] {{ color: {p['ok']}; border: 1px solid {p['ok']}; background: transparent; }}
    QLabel#Pill[pill="rerun"] {{ color: {p['warn']}; border: 1px solid {p['warn']}; background: transparent; }}

    /* buttons */
    QPushButton {{ background: {p['surface2']}; border: 1px solid {p['border']};
        border-radius: 8px; padding: 7px 13px; color: {p['text']}; }}
    QPushButton:hover {{ background: {p['surface3']}; border-color: {p['accent']}; }}
    QPushButton:pressed {{ background: {p['surface3']}; }}
    QPushButton:disabled {{ color: {p['faint']}; border-color: {p['border_soft']}; }}
    QPushButton#Accent {{ background: {p['accent']}; border: none; color: #FFFFFF; font-weight: 600;
        font-family: {MONO}; padding: 8px 18px; }}
    QPushButton#Accent:hover {{ background: {p['accent_press']}; }}
    QPushButton#Accent:disabled {{ background: {p['surface3']}; color: {p['faint']}; }}
    QPushButton#Accent[stale="true"]:enabled {{ background: {p['warn']}; color: #1A1206; }}
    QPushButton#Accent[stale="true"]:enabled:hover {{ background: {p['warn']}; }}
    /* a subtle fold/unfold header (Collapsible) — reads as a section sub-label,
       not a full button */
    QToolButton#Collapse {{ border: none; background: transparent; color: {p['muted']};
        font-family: {MONO}; font-size: 11px; letter-spacing: 1px; padding: 3px 1px; }}
    QToolButton#Collapse:hover {{ color: {p['text']}; }}
    QToolButton#Collapse:checked {{ color: {p['text']}; }}
    QPushButton#Seg {{ padding: 4px 10px; border-radius: 6px; }}
    QPushButton#Seg:checked {{ background: {p['accent']}; border-color: {p['accent']}; color: #FFFFFF; }}
    QPushButton#Seg:checked:hover {{ background: {p['accent_press']}; }}
    QPushButton#Toggle:checked {{ background: {p['accent']}; border-color: {p['accent']};
        color: #FFFFFF; font-weight: 600; }}
    QPushButton#Toggle:checked:hover {{ background: {p['accent_press']}; }}
    /* Compare tab — one card per model. The card IS the click target for that
       model's settings, so it has to read as pressable and has to show which one
       the panel is on: [editing] is now the ONLY thing that says so (the "Edit
       settings ▸" link that used to say it in words is gone). Filled, not dashed —
       a hairline is not enough to carry that alone. */
    QFrame#ModelCard {{ background: {p['surface']}; border: 1px solid {p['border_soft']};
        border-radius: 10px; }}
    QFrame#ModelCard:hover {{ border: 1px solid {p['faint']}; }}
    QFrame#ModelCard[editing="true"] {{ background: {p['sel']};
        border: 1px solid {p['accent']}; }}
    /* [shown] is a different question — which result the OTHER tabs are displaying —
       so it gets a different mark, and the two can be true at once. */
    QFrame#ModelCard[shown="true"] {{ border-left: 3px solid {p['accent2']}; }}
    QFrame#CardRule {{ background: {p['border_soft']}; border: none; }}
    /* NOTE: scoped to #CardTitle, never bare QCheckBox — ToggleSwitch subclasses
       QCheckBox and paints itself, and every param toggle in a card is one. */
    QCheckBox#CardTitle {{ font-family: {MONO}; font-weight: 600; font-size: 13px; spacing: 8px; }}
    QCheckBox#CardTitle::indicator {{ width: 15px; height: 15px; border-radius: 4px;
        border: 1px solid {p['border']}; background: {p['surface2']}; }}
    QCheckBox#CardTitle::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}
    QCheckBox#CardTitle::indicator:disabled {{ border-color: {p['border_soft']}; }}

    /* inputs */
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{ background: {p['surface2']};
        border: 1px solid {p['border']}; border-radius: 7px; padding: 5px 8px;
        selection-background-color: {p['accent']}; selection-color: #fff; }}
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {p['accent']}; }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
    QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox::down-arrow {{ image: none; border-left: 4px solid transparent;
        border-right: 4px solid transparent; border-top: 5px solid {p['muted']}; margin-right: 8px; }}
    QComboBox QAbstractItemView {{ background: {p['surface2']}; border: 1px solid {p['border']};
        selection-background-color: {p['accent']}; selection-color: #fff; outline: none; }}

    /* tables (Repeatability log) */
    QTableWidget {{ background: {p['surface']}; border: 1px solid {p['border']};
        border-radius: 8px; gridline-color: {p['border_soft']}; }}
    QTableWidget::item {{ padding: 3px 7px; color: {p['text']}; }}
    QTableWidget#StatTable {{ font-family: {MONO}; }}
    QHeaderView::section {{ background: {p['surface2']}; color: {p['muted']};
        border: none; border-bottom: 1px solid {p['border']}; padding: 5px 8px;
        font-family: {MONO}; font-size: 11px; }}
    QTableCornerButton::section {{ background: {p['surface2']}; border: none; }}

    /* batch preview list */
    QListWidget#BatchPreview {{ background: {p['surface']}; border: 1px solid {p['border']};
        border-radius: 8px; font-family: {MONO}; font-size: 11px; color: {p['text']}; padding: 3px; }}
    QListWidget#BatchPreview::item {{ padding: 3px 7px; border-radius: 4px; }}
    QListWidget#BatchPreview::item:selected {{ background: {p['surface3']}; color: {p['text']}; }}

    /* sliders */
    QSlider::groove:horizontal {{ height: 5px; background: {p['surface3']}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: #FFFFFF; border: 2px solid {p['accent']};
        width: 13px; height: 13px; margin: -5px 0; border-radius: 8px; }}
    QSlider::handle:horizontal:hover {{ border-color: {p['accent_press']}; }}

    /* group boxes */
    QGroupBox {{ border: 1px solid {p['border_soft']}; border-radius: 10px; margin-top: 8px;
        padding: 14px 12px 12px; background: {p['surface']}; }}

    /* image drop tiles */
    QFrame#ImageDrop {{ border: 1.5px dashed {p['border']}; border-radius: 9px; background: {p['surface2']}; }}
    QFrame#ImageDrop:hover {{ border-color: {p['accent']}; }}
    QLabel#Thumb {{ background: {p['bg']}; border-radius: 5px; }}

    /* tabs */
    QTabWidget::pane {{ border: none; background: {p['bg']}; }}
    QTabBar {{ background: {p['surface2']}; }}
    QTabBar::tab {{ font-family: {MONO}; font-size: 12px; color: {p['muted']};
        background: {p['surface2']}; padding: 8px 15px; margin: 4px 1px 0; border: 1px solid transparent;
        border-top-left-radius: 8px; border-top-right-radius: 8px; }}
    QTabBar::tab:selected {{ color: {p['text']}; background: {p['bg']};
        border-color: {p['border']}; border-bottom-color: {p['bg']}; font-weight: 600; }}
    QTabBar::tab:hover:!selected {{ color: {p['text']}; }}

    /* toolbar + status + menu */
    QToolBar {{ background: {p['surface2']}; border-bottom: 1px solid {p['border']}; spacing: 8px; padding: 7px 12px; }}
    QStatusBar {{ background: {p['surface2']}; border-top: 1px solid {p['border']};
        font-family: {MONO}; font-size: 11px; color: {p['muted']}; }}
    QStatusBar::item {{ border: none; }}
    QMenuBar {{ background: {p['surface2']}; color: {p['text']}; }}
    QMenuBar::item:selected {{ background: {p['surface3']}; }}
    QMenu {{ background: {p['surface2']}; border: 1px solid {p['border']}; padding: 4px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {p['accent']}; color: #fff; }}

    /* scrollbars */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p['surface3']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::handle:vertical:hover {{ background: {p['muted']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {p['surface3']}; border-radius: 5px; min-width: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollArea {{ border: none; background: transparent; }}

    QToolTip {{ background: {p['surface3']}; color: {p['text']};
        border: 1px solid {p['border']}; border-radius: 6px; padding: 5px 8px; }}
    QProgressBar {{ border: none; background: {p['surface3']}; border-radius: 3px; height: 4px; text-align: center; }}
    QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}
    """


def apply_theme(app: QApplication, name: str = "dark") -> dict:
    p = PALETTES.get(name, DARK)
    app.setStyle("Fusion")
    app.setPalette(_qpalette(p))
    app.setStyleSheet(_qss(p))
    return p
