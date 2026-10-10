from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

LABEL_COLS = ["stage", "spi3_stage"]


def spi3_to_stage(spi3: float) -> int:
    """SPI-3 → drought stage label using the spec's SPI classes."""
    if pd.isna(spi3):
        return -1
    if spi3 <= -2.0:
        return 4
    if spi3 <= -1.5:
        return 3
    if spi3 <= -1.0:
        return 2
    if spi3 <= -0.99:
        return 1
    return 0


def train_drought_classifier(
    df_stages: pd.DataFrame,
    config: Dict[str, Any],
    telemetry: Any,
    logger: logging.Logger,
    force: bool = False
) -> Tuple[RandomForestClassifier, pd.DataFrame]:
    """
    Trains a multi-output Random Forest predicting TWO labels per tehsil-month:
      - stage:       composite CDI label (0 No drought … 4 Extreme)
      - spi3_stage:  SPI-3 meteorological label (spec §4 classes)
    Features are the observables only (SPI-1/3/6/12, SSMI, NDVI, EVI, PCI) —
    CDI/VCI are never used as features, so predictions stay independent of the
    label. Train/test split is by year to avoid temporal leakage.
    Saves: model, per-label classification reports, confusion matrix, importance.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / "outputs" / "models"
    figs_dir = project_root / "outputs" / "figures"
    csv_dir = project_root / "outputs" / "csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "drought_rf_model.joblib"
    report_path = csv_dir / "rf_classification_report.csv"

    if model_path.exists() and report_path.exists() and not force:
        logger.info(f"[CACHE] Found existing RF model at {model_path}")
        model = joblib.load(model_path)
        report_df = pd.read_csv(report_path)
        telemetry.add_output(model_path)
        telemetry.add_output(report_path)
        return model, report_df

    logger.info("Training Random Forest drought stage classifier "
                "(labels: stage + spi3_stage; predictions from index features only)...")

    feature_cols = ["SPI-1", "SPI-3", "SPI-6", "SPI-12", "SSMI"]
    for sat_col in ["NDVI", "EVI", "PCI"]:
        if sat_col in df_stages.columns and df_stages[sat_col].notna().sum() > 0.1 * len(df_stages):
            feature_cols.append(sat_col)

    cols = list(dict.fromkeys(feature_cols + ["stage", "SPI-3", "month"]))
    df_ml = df_stages[cols].copy()
    df_ml["spi3_stage"] = df_ml["SPI-3"].apply(spi3_to_stage)
    df_ml = df_ml.dropna(subset=feature_cols + ["stage"])
    df_ml = df_ml[df_ml["spi3_stage"] >= 0]
    if len(df_ml) < 50:
        logger.warning("Insufficient labeled data for RF training (< 50 rows). Skipping.")
        return None, pd.DataFrame()

    df_ml["year"] = pd.to_datetime(df_ml["month"]).dt.year
    split_year = df_ml["year"].quantile(0.8).astype(int)
    logger.info(f"Train/test split: train <= {split_year}, test > {split_year}")

    train = df_ml[df_ml["year"] <= split_year]
    test = df_ml[df_ml["year"] > split_year]

    X_train = train[feature_cols]
    y_train = train[LABEL_COLS]
    X_test = test[feature_cols]
    y_test = test[LABEL_COLS]

    if len(y_test) == 0:
        logger.warning("Test set is empty after year-based split. Using 20% random holdout.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df_ml[feature_cols], df_ml[LABEL_COLS],
            test_size=config["ml"]["test_size"],
            random_state=config["ml"]["random_state"]
        )

    rf = RandomForestClassifier(
        n_estimators=config["ml"]["n_estimators"],
        max_depth=config["ml"]["max_depth"],
        random_state=config["ml"]["random_state"],
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_pred = np.atleast_2d(y_pred)
    if y_pred.shape[1] != len(LABEL_COLS):  # single-column edge case
        y_pred = y_pred.reshape(len(y_pred), -1)

    reports = {}
    for i, label in enumerate(LABEL_COLS):
        acc = accuracy_score(y_test[label], y_pred[:, i])
        f1 = f1_score(y_test[label], y_pred[:, i], average="macro", zero_division=0)
        reports[label] = acc
        logger.info(f"RF [{label}] accuracy: {acc:.3f}, macro-F1: {f1:.3f} "
                    f"(label = ground truth, prediction = RF output)")
        rep = classification_report(y_test[label], y_pred[:, i],
                                    output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).transpose()
        out_path = csv_dir / f"rf_classification_report{'' if i == 0 else '_' + label}.csv"
        rep_df.to_csv(out_path)
        telemetry.add_output(out_path)

    report_df = pd.read_csv(report_path)

    # ── Confusion matrices (one per label) ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, (label, ax) in enumerate(zip(LABEL_COLS, axes)):
        labels = sorted(set(y_test[label]) | set(y_pred[:, i]))
        cm = confusion_matrix(y_test[label], y_pred[:, i], labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=[f"S{l}" for l in labels],
                    yticklabels=[f"S{l}" for l in labels])
        acc = reports[label]
        ax.set_title(f"{label} (accuracy {acc:.3f})", fontsize=11, fontweight="bold")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    fig.suptitle("Random Forest – Drought Stage Confusion Matrices\n"
                 "S0 No drought · S1 Mild · S2 Moderate · S3 Severe · S4 Extreme",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    cm_path = figs_dir / "rf_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    telemetry.add_output(cm_path)

    # ── Feature importance ─────────────────────────────────────────────────
    fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    fi.plot(kind="barh", color="steelblue", edgecolor="navy")
    plt.title("Random Forest – Feature Importance\n"
              "(features are predictions' inputs; CDI/VCI are labels, never features)",
              fontsize=11, fontweight="bold")
    plt.xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    fi_path = figs_dir / "rf_feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    telemetry.add_output(fi_path)

    joblib.dump(rf, model_path)
    telemetry.add_output(model_path)
    logger.info(f"RF model saved to {model_path}. "
                + ", ".join(f"{k} acc {v:.3f}" for k, v in reports.items()))
    return rf, report_df
