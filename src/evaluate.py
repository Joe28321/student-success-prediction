"""Evaluation helpers for the Student Success Predictor.

- compute_metrics: accuracy / precision / recall / F1 / confusion matrix.
  'at_risk' is always the positive class (pos_label handling is explicit).
- roc_auc_summary: ROC-AUC from predicted probabilities (only when both
  classes are present and the pipeline supports predict_proba).
- make_all_figures: comparison bar chart, confusion matrices, ROC curves,
  class distribution plots - all saved to results/figures/.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ACCENT = "#2A9D8F"      # teal
ACCENT2 = "#264653"     # dark navy
RED = "#E76F51"


def compute_metrics(y_true, y_pred, pipeline=None, X_test=None):
    """Test-set metrics; positive class = at_risk.

    Returns a dict of floats (and a serialisable confusion matrix).
    """
    labels = config.TARGET_LABELS  # ["not_at_risk", "at_risk"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(
            precision_score(y_true, y_pred, pos_label="at_risk",
                            zero_division=0)
        ),
        "recall_positive": float(
            recall_score(y_true, y_pred, pos_label="at_risk",
                         zero_division=0)
        ),
        "f1_positive": float(
            f1_score(y_true, y_pred, pos_label="at_risk", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
        },
        "n_test": int(len(y_true)),
    }
    return metrics


def roc_auc_summary(results_df, model_blobs, X_test, y_test):
    """Add roc_auc to the results frame (in place).

    Handles the string labels explicitly: we find the column of
    predict_proba that belongs to 'at_risk' and compare against a binary
    version of y_test (avoids sklearn's implicit pos_label resolution).
    """
    at_risk = config.POSITIVE_CLASS
    two_classes = y_test.nunique() == 2
    y_bin = (y_test == at_risk).astype(int)
    for i, row in results_df.iterrows():
        name = row["model"]
        pipeline = model_blobs[name]
        try:
            proba = pipeline.predict_proba(X_test)  # columns follow classes_
            col = list(pipeline.classes_).index(at_risk)
            if not two_classes or col < 0:
                metrics_row = {"roc_auc": None}
            else:
                metrics_row = {"roc_auc": float(roc_auc_score(y_bin, proba[:, col]))}
        except Exception:
            metrics_row = {"roc_auc": None}
        for k, v in metrics_row.items():
            results_df.loc[i, k] = v


def _save(fig, filename):
    path = config.FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] figure saved: {path.name}")


def plot_metric_comparison(results_df):
    """Grouped bar chart of the main test metrics per model."""
    metrics_cols = [
        ("accuracy", "Accuracy"),
        ("precision_positive", "Precision (at_risk)"),
        ("recall_positive", "Recall (at_risk)"),
        ("f1_positive", "F1 (at_risk)"),
    ]
    if "roc_auc" in results_df.columns:
        metrics_cols.append(("roc_auc", "ROC-AUC"))

    n_models = len(results_df)
    x = np.arange(len(metrics_cols))
    width = 0.8 / max(n_models, 1)
    colors = [ACCENT2, ACCENT, RED]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for idx, (_, row) in enumerate(results_df.iterrows()):
        vals = [
            float(row[c]) if row[c] is not None and not (
                isinstance(row[c], float) and np.isnan(row[c])
            ) else 0.0
            for c, _ in metrics_cols
        ]
        bars = ax.bar(x + idx * width, vals, width,
                      label=row["model"], color=colors[idx % len(colors)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([label for _, label in metrics_cols], fontsize=9)
    ax.set_ylabel("Score (higher is better)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model comparison on the held-out test split "
                 "(positive class = at_risk)")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "model_comparison.png")


def plot_confusion_matrices(results_df):
    """One confusion-matrix panel per model."""
    fig, axes = plt.subplots(1, len(results_df),
                             figsize=(4.6 * len(results_df), 4.2))
    if len(results_df) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, results_df.iterrows()):
        cm = np.array(row["confusion_matrix"]["matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(config.TARGET_LABELS, fontsize=8)
        ax.set_yticklabels(config.TARGET_LABELS, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(row["model"], fontsize=10)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, "confusion_matrices.png")


def plot_roc_curves(results_df, model_blobs, X_test, y_test):
    """ROC curves for all models that support predict_proba."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plotted = False
    for (_, row), color in zip(
        results_df.iterrows(), [ACCENT2, ACCENT, RED]
    ):
        name = row["model"]
        if row.get("roc_auc") is None:
            continue
        proba = model_blobs[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba, pos_label="at_risk")
        ax.plot(fpr, tpr, label=f"{name} (AUC={row['roc_auc']:.3f})",
                color=color)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - test split (positive class = at_risk)")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "roc_curves.png")


def plot_class_distribution(df):
    """Target class distribution + G3 histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    counts = df["target"].value_counts().reindex(config.TARGET_LABELS)
    bars = axes[0].bar(counts.index, counts.values,
                       color=[ACCENT, RED])
    axes[0].bar_label(bars, fmt="%d")
    axes[0].set_title("Class distribution (target)")
    axes[0].set_ylabel("Students")
    total = counts.sum()
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + total * 0.02, f"{v / total:.0%}",
                     ha="center", fontsize=9)

    g3_counts = df["G3"].value_counts().sort_index()
    axes[1].bar(g3_counts.index, g3_counts.values, color=ACCENT2)
    axes[1].axvline(config.G3_PASS_THRESHOLD - 0.5, color=RED,
                    linestyle="--", label=f"risk threshold G3 < {config.G3_PASS_THRESHOLD}")
    axes[1].set_title("Final grade G3 distribution")
    axes[1].set_xlabel("G3 (0-20)")
    axes[1].set_ylabel("Students")
    axes[1].legend(fontsize=8)
    _save(fig, "class_distribution.png")


def plot_feature_importance_tr(model, feature_names_out, importance_values,
                               title, filename, top_n=15):
    """Horizontal bar chart of tree feature importances."""
    order = np.argsort(importance_values)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh([feature_names_out[i] for i in order][::-1],
            importance_values[order][::-1], color=ACCENT)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Importance")
    _save(fig, filename)


def make_all_figures(results_df, model_blobs, X_test, y_test, df, audit,
                     perm_summary):
    """Generate every figure used by the app and the presentation."""
    plot_metric_comparison(results_df)
    plot_confusion_matrices(results_df)
    try:
        plot_roc_curves(results_df, model_blobs, X_test, y_test)
    except Exception as exc:
        print(f"[evaluate] ROC plot skipped: {exc}")
    plot_class_distribution(df)

    # Tree-based feature importance of the best model.
    best = results_df.iloc[0]["model"]
    pipeline = model_blobs[best]
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        plot_feature_importance_tr(
            clf, np.array(names), np.array(clf.feature_importances_),
            f"Feature importance - {best} (test split)",
            "feature_importance.png",
        )


def permutation_importance_summary(pipeline, X_test, y_test, n_repeats=10):
    """Permutation importance on the HELD-OUT test split (original features)."""
    from sklearn.inspection import permutation_importance as _pi
    from sklearn.metrics import f1_score, make_scorer

    scorer = make_scorer(f1_score, pos_label=config.POSITIVE_CLASS,
                         zero_division=0)
    result = _pi(
        pipeline, X_test, y_test, scoring=scorer, n_repeats=n_repeats,
        random_state=config.RANDOM_SEED, n_jobs=1,
    )
    importances = result.importances_mean
    order = np.argsort(importances)[::-1]
    top = [
        {"feature": str(X_test.columns[i]),
         "importance_mean": round(float(importances[i]), 4),
         "importance_std": round(float(result.importances_std[i]), 4)}
        for i in order if importances[i] > 0
    ][:15]
    return {
        "scoring": "f1 (positive class = at_risk)",
        "n_repeats": n_repeats,
        "note": ("computed on the held-out test split; associations only, "
                 "NOT proof of causation"),
        "top_features": top,
    }
