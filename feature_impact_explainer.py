#!/usr/bin/env python3
"""Train and evaluate an SVM classifier from synthetic_similarity_labeled_1000.csv."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "synthetic_similarity_labeled_1000.csv"
MODEL_PATH = ROOT / "models" / "global_svm_model.pkl"

FEATURES = [
    "G0_x", "G0_y", "G1_x", "G1_y", "G2_x", "G2_y", "G3_x", "G3_y",
    "H0_x", "H0_y", "H1_x", "H1_y", "H2_x", "H2_y", "H3_x", "H3_y",
]
TARGET = "predicted_label"


def global_model() -> None:
    df = pd.read_csv(CSV_PATH)

    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_samples = pd.concat([df[df['predicted_label']==0].sample(n=100), df[df['predicted_label']==1].sample(n=100)], axis=0)

    x = df_samples[FEATURES]
    y = df_samples[TARGET].astype(int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(x_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(x_test)

    test_acc = float(accuracy_score(y_test, y_pred))
    train_acc = float(model.score(x_train, y_train))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    return model

def explain_model(model, x_train, y_train):
    explainer = LimeTabularExplainer(
        training_data=x_train.values,
        feature_names=FEATURES,
        class_names=['0', '1'],
        mode='classification'
    )

    return explainer
def explain_prediction(explainer, model, x_i):
    x_i = np.asarray(x_i, dtype=float)

    # LIME calls predict_fn with numpy arrays; rebuild a DataFrame to preserve feature names.
    def predict_fn(x_batch):
        x_df = pd.DataFrame(x_batch, columns=FEATURES)
        return model.predict_proba(x_df)

    pred_label_idx = int(np.argmax(predict_fn(x_i.reshape(1, -1))[0]))

    exp = explainer.explain_instance(
        data_row=x_i,
        predict_fn=predict_fn,
        num_features=len(FEATURES),
        labels=(pred_label_idx,),
    )

    # Build a stable, full-length impact vector in the original FEATURES order.
    impact_values = [0.0] * len(FEATURES)
    for feature_idx, weight in exp.local_exp.get(pred_label_idx, []):
        impact_values[int(feature_idx)] = float(weight)

    return impact_values


if __name__ == "__main__":
    model = global_model()

    df = pd.read_csv(CSV_PATH)
    x_train = df[FEATURES]
    y_train = df[TARGET].astype(int)
    explainer = explain_model(model, x_train, y_train)
    impact_values = explain_prediction(explainer, model, df[FEATURES].iloc[0].to_numpy())
    print(f"Feature impact values for first sample: {impact_values}")