"""Print the measurement sites saved from the GUI (and the ROI/Δ beside them).

The app stores what you marked in QSettings; this reads the same keys so a
headless study — or a second pair of eyes — sees exactly what you placed.

    .venv/bin/python tools/show_sites.py
"""
import json
import sys


def load() -> dict:
    from PySide6.QtCore import QSettings
    s = QSettings("FSStudio", "FoundationStereoStudio")
    def blob(k, d):
        try:
            return json.loads(s.value(k, "") or json.dumps(d))
        except (ValueError, TypeError):
            return d
    return {"sites": blob("study_sites", []), "roi": blob("roi", {}),
            "boxes": blob("box_presets", {})}


def main() -> int:
    d = load()
    roi = d["roi"] or {}
    print(f"ROI   : {roi.get('roi')}   Δ {roi.get('shift', 0)}")
    sites = d["sites"]
    if not sites:
        print("sites : none marked yet — tick 'Mark: pin' on the Input tab and click the pins")
        return 0
    pins = [s for s in sites if s["kind"] == "pin"]
    refs = [s for s in sites if s["kind"] == "ref"]
    print(f"sites : {len(pins)} pin(s), {len(refs)} reference(s)")
    for s in sites:
        print(f"  {s['kind']:4s} {s['name']:10s} ({s['x']:5d}, {s['y']:5d})")
    for p in pins:                       # the pairing the study will use
        if refs:
            r = min(refs, key=lambda q: (q["x"]-p["x"])**2 + (q["y"]-p["y"])**2)
            dist = ((r["x"]-p["x"])**2 + (r["y"]-p["y"])**2) ** 0.5
            print(f"  -> {p['name']} measured against {r['name']} ({dist:.0f} px away)")
        else:
            print(f"  -> {p['name']} has NO reference — mark one near it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
