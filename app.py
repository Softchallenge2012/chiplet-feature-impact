#!/usr/bin/env python3
"""Flask app to view layout_config values and LIME feature impacts."""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template

from feature_impact_explainer import FEATURES, CSV_PATH, TARGET, explain_model, explain_prediction, global_model

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

app = Flask(__name__)

HEAT_FILES = sorted([p.name for p in DATA_DIR.glob("heat_analysis_v*.json")])[:5]

# Train the model once and initialize a reusable explainer.
MODEL = global_model()
TRAIN_DF = None
try:
    import pandas as pd

    TRAIN_DF = pd.read_csv(CSV_PATH)
except Exception:
    TRAIN_DF = None

if TRAIN_DF is None:
    raise RuntimeError(f"Unable to read training CSV: {CSV_PATH}")

X_TRAIN = TRAIN_DF[FEATURES]
Y_TRAIN = TRAIN_DF[TARGET].astype(int)
EXPLAINER = explain_model(MODEL, X_TRAIN, Y_TRAIN)


def _load_layout_config(json_name: str) -> dict[str, list[float]]:
    json_path = DATA_DIR / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_name}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload or "layout_config" not in payload[0]:
            raise ValueError(f"layout_config missing in {json_name}")
        return payload[0]["layout_config"]
    if isinstance(payload, dict) and "layout_config" in payload:
        return payload["layout_config"]

    raise ValueError(f"Unsupported JSON layout format in {json_name}")


def _layout_to_feature_row(layout_config: dict[str, list[float]]) -> list[float]:
    values: list[float] = []
    for feature in FEATURES:
        die, axis = feature.split("_")
        coords = layout_config[die]
        values.append(float(coords[0] if axis == "x" else coords[1]))
    return values


def _switched_absolute_impacts(features: list[str], impacts: list[float]) -> tuple[list[str], list[float]]:
    """Switch x/y per die for charting and return absolute impact values."""
    switched_features: list[str] = []
    switched_impacts: list[float] = []

    # Keep behavior stable even if upstream returns shorter/longer impact vectors.
    normalized_impacts = list(impacts[: len(features)])
    if len(normalized_impacts) < len(features):
        normalized_impacts.extend([0.0] * (len(features) - len(normalized_impacts)))

    # FEATURES are expected in die pairs: *_x followed by *_y.
    for i in range(0, len(features), 2):
        fx = features[i]
        fy = features[i + 1]
        ix = normalized_impacts[i]
        iy = normalized_impacts[i + 1]

        die_x = fx.rsplit("_", 1)[0]
        die_y = fy.rsplit("_", 1)[0]
        if die_x != die_y:
            raise ValueError(f"Unexpected feature pairing: {fx}, {fy}")

        switched_features.extend([f"{die_x}_height", f"{die_x}_width"])
        switched_impacts.extend([abs(float(iy)), abs(float(ix))])

    return switched_features, switched_impacts


@app.route("/")
def index():
    return render_template("index.html", files=HEAT_FILES)


@app.route("/api/layout/<filename>")
def get_layout_and_impacts(filename: str):
    if filename not in HEAT_FILES:
        return jsonify({"error": "Invalid file selection"}), 400

    try:
        layout_config = _load_layout_config(filename)
        feature_row = _layout_to_feature_row(layout_config)
        impact_values = explain_prediction(EXPLAINER, MODEL, feature_row)
        chart_features, chart_impact_values = _switched_absolute_impacts(FEATURES, impact_values)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "filename": filename,
            "layout_config": layout_config,
            "features": chart_features,
            "impact_values": chart_impact_values,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
