"""Definisi model dan metrik yang identik untuk CV dan uji eksternal."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pipeline_config import RANDOM_SEED


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
                ),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def model_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def batik_score(model, values):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(values)[:, 1]
    decision = np.clip(np.asarray(model.decision_function(values), float), -50, 50)
    return 1.0 / (1.0 + np.exp(-decision))


def metric_values(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "recall_non_batik": recall_score(
            y_true, y_pred, labels=[0], average="macro", zero_division=0
        ),
        "recall_batik": recall_score(
            y_true, y_pred, labels=[1], average="macro", zero_division=0
        ),
    }


def per_class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return pd.DataFrame(
        [
            {
                "kelas": class_name,
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index, class_name in enumerate(["non_batik", "batik"])
        ]
    )
