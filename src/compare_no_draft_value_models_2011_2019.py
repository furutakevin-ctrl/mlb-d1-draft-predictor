"""Compare no-draft-slot models and tune college-stat emphasis.

Draft pick is excluded from model inputs. It is used only to define a
later-half evaluation subset within each draft year. Outer folds estimate
out-of-fold performance; inner folds select a college-stat group weight using
average precision among later picks.
"""
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2011_2019.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "no_draft_value_models_2011_2019.txt"
PREDICTION_PATH = PROJECT_ROOT / "data" / "processed" / "no_draft_value_oof_predictions_2011_2019.csv"
METRICS_PATH = REPORT_DIR / "no_draft_value_model_metrics_2011_2019.csv"
WEIGHT_PATH = REPORT_DIR / "college_stat_weight_selection_2011_2019.csv"

TARGET = "made_mlb_ever"
COLLEGE_FEATURES = [
    "college_PA", "college_OPS", "college_BB_rate", "college_K_rate", "college_HR_rate",
]
CONTEXT_FEATURES = ["age_at_draft", "team_SOS"]
WEIGHT_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


class GroupWeight(BaseEstimator, TransformerMixin):
    """Multiply a standardized feature group by a tunable factor."""

    def __init__(self, weight=1.0):
        self.weight = weight

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * float(self.weight)


def position_family(values):
    values = values.astype("string").str.strip()
    values = values.mask(values.isin(["CF", "LF", "RF", "OF"]), "OF")
    values = values.mask(values.isin(["1B", "2B", "3B", "SS", "IF"]), "IF")
    return values


def make_pipeline(include_position):
    college_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("group_weight", GroupWeight()),
        ]
    )
    context_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transforms = [
        ("college", college_pipeline, COLLEGE_FEATURES),
        ("context", context_pipeline, CONTEXT_FEATURES),
    ]
    feature_columns = [*COLLEGE_FEATURES, *CONTEXT_FEATURES]
    if include_position:
        position_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transforms.append(("position", position_pipeline, ["position_family"]))
        feature_columns.append("position_family")

    preprocessor = ColumnTransformer(transformers=transforms, remainder="drop")
    estimator = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0)),
        ]
    )
    return estimator, feature_columns


def later_pick_average_precision(estimator, X_validation, y_validation):
    """Inner-CV scorer. The pick-derived flag selects the evaluation rows only."""
    mask = X_validation["later_half_by_pick"].to_numpy(dtype=bool)
    if not mask.any():
        return 0.0
    y_array = np.asarray(y_validation, dtype=int)
    if y_array[mask].sum() == 0:
        return 0.0
    probabilities = estimator.predict_proba(X_validation)[:, 1]
    return average_precision_score(y_array[mask], probabilities[mask])


def score_subset(y_true, probability):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.001, 0.999)
    if len(y_true) == 0:
        return {"n": 0, "positives": 0}

    positives = int(y_true.sum())
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, probability)
    else:
        auc = np.nan
    average_precision = average_precision_score(y_true, probability) if positives else np.nan
    top_n = max(1, ceil(0.10 * len(y_true)))
    top_indices = np.argsort(probability)[::-1][:top_n]
    top_rate = float(y_true[top_indices].mean())
    base_rate = float(y_true.mean())
    return {
        "n": len(y_true),
        "positives": positives,
        "roc_auc": auc,
        "average_precision": average_precision,
        "brier_score": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, probability, labels=[0, 1]),
        "top_10pct_mlb_rate": top_rate,
        "top_10pct_lift": top_rate / base_rate if base_rate else np.nan,
        "top_10pct_recall": float(y_true[top_indices].sum() / positives) if positives else np.nan,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing combined modeling table: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    required = {
        TARGET, "draft_year", "overall_pick", "position", *COLLEGE_FEATURES,
        *CONTEXT_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Combined table is missing columns: {missing}")

    for column in [TARGET, "draft_year", "overall_pick", *COLLEGE_FEATURES, *CONTEXT_FEATURES]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[[TARGET, "draft_year", "overall_pick", *CONTEXT_FEATURES]].isna().any().any():
        raise ValueError("Missing outcome, draft-year/pick key, age, or SOS values.")
    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1.")
    if df.duplicated(["draft_year", "overall_pick"]).any():
        raise ValueError("Duplicate draft-year/pick keys found.")
    if df["position"].isna().any():
        raise ValueError("Missing positions found.")

    df["position_family"] = position_family(df["position"])
    pick_rank = df.groupby("draft_year")["overall_pick"].rank(method="first", pct=True)
    df["later_half_by_pick"] = pick_rank.gt(0.5)

    # This X includes the pick-derived flag only so the inner-CV scorer can
    # evaluate later selections. ColumnTransformer drops it before fitting.
    X = df[[*COLLEGE_FEATURES, *CONTEXT_FEATURES, "position_family"]].copy()
    X["later_half_by_pick"] = df["later_half_by_pick"].to_numpy()
    y = df[TARGET].astype(int).to_numpy()
    if set(X.columns) & {"overall_pick", "draft_round"}:
        raise RuntimeError("Draft position was included among model inputs.")

    configurations = [
        ("college_age_sos_no_position_equal", False, False),
        ("college_age_sos_with_position_equal", True, False),
        ("college_age_sos_no_position_tuned", False, True),
        ("college_age_sos_with_position_tuned", True, True),
    ]
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prediction_rows = []
    weight_rows = []

    for fold_number, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fold_number)

        for model_name, include_position, tune_weight in configurations:
            estimator, feature_columns = make_pipeline(include_position)
            # Check that pick order is only used by the evaluator, never as a predictor.
            if "overall_pick" in feature_columns or "draft_round" in feature_columns:
                raise RuntimeError("Draft position leaked into model features.")
            if tune_weight:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid={"preprocess__college__group_weight__weight": WEIGHT_GRID},
                    scoring=later_pick_average_precision,
                    cv=inner_cv,
                    refit=True,
                    n_jobs=None,
                    error_score="raise",
                )
                search.fit(X_train, y_train)
                fitted = search.best_estimator_
                selected_weight = float(search.best_params_["preprocess__college__group_weight__weight"])
                for candidate_weight, candidate_score in zip(
                    search.cv_results_["param_preprocess__college__group_weight__weight"],
                    search.cv_results_["mean_test_score"],
                ):
                    weight_rows.append(
                        {
                            "model": model_name,
                            "outer_fold": fold_number,
                            "college_stat_weight": float(candidate_weight),
                            "inner_later_half_average_precision": float(candidate_score),
                            "selected_in_outer_fold": float(candidate_weight) == selected_weight,
                        }
                    )
            else:
                selected_weight = 1.0
                estimator.set_params(preprocess__college__group_weight__weight=selected_weight)
                fitted = estimator.fit(X_train, y_train)

            probability = np.clip(fitted.predict_proba(X_test)[:, 1], 0.001, 0.999)
            for row_index, predicted_probability in zip(test_index, probability):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "draft_year": int(df.iloc[row_index]["draft_year"]),
                        "player_name": df.iloc[row_index]["player_name"],
                        "overall_pick": int(df.iloc[row_index]["overall_pick"]),
                        "later_half_by_pick": bool(df.iloc[row_index]["later_half_by_pick"]),
                        TARGET: int(y[row_index]),
                        "oof_probability": float(predicted_probability),
                        "college_stat_weight": selected_weight,
                        "outer_fold": fold_number,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        for segment, mask in [
            ("all_candidates", pd.Series(True, index=group.index)),
            ("later_half_by_pick_within_year", group["later_half_by_pick"]),
        ]:
            scored = score_subset(group.loc[mask, TARGET], group.loc[mask, "oof_probability"])
            metric_rows.append({"model": model_name, "evaluation_group": segment, **scored})

    metrics = pd.DataFrame(metric_rows)
    weight_selection = pd.DataFrame(weight_rows)
    selected_weights = (
        weight_selection[weight_selection["selected_in_outer_fold"]]
        .groupby("model")["college_stat_weight"]
        .agg(lambda values: ", ".join(f"{value:g}" for value in values))
        .to_dict()
    )

    year_summary = (
        df.groupby("draft_year")
        .agg(
            players=("player_name", "size"),
            positive_outcomes=(TARGET, "sum"),
            later_half_players=("later_half_by_pick", "sum"),
        )
        .reset_index()
    )
    report_lines = [
        "NO-DRAFT-POSITION VALUE MODEL COMPARISON, 2011-2019",
        "=" * 54,
        "",
        f"Players: {len(df)}",
        f"Positive outcomes: {int(y.sum())} ({y.mean():.1%})",
        "Model inputs exclude overall_pick and draft_round.",
        "Later-half status is used only for evaluation and inner-fold weight selection.",
        "",
        "College stat emphasis factors tested after standardization:",
        ", ".join(f"{weight:g}" for weight in WEIGHT_GRID),
        "",
        "Model metrics:",
        metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        "Selected college-stat weight by outer fold for tuned models:",
        "\n".join(f"{model}: {values}" for model, values in selected_weights.items()),
        "",
        "Draft-year coverage:",
        year_summary.to_string(index=False),
        "",
        "How to read the later-pick metrics:",
        "Within each draft year, later-half candidates are the half with higher overall_pick values.",
        "The model does not receive overall_pick. Top-10% rate and lift show whether its highest-ranked later picks outperform the later-half base rate.",
        "",
        "Validation note:",
        "Weight selection uses inner-fold average precision among later picks; reported model performance uses outer-fold predictions.",
        "Outer folds randomly mix draft years, so this estimates pooled performance, not performance on a future draft class.",
        "The cohort remains provisional while school-match review rows are unresolved.",
    ]
    report = "\n".join(report_lines)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    weight_selection.to_csv(WEIGHT_PATH, index=False)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {REPORT_PATH}")
    print(f"Saved out-of-fold predictions to: {PREDICTION_PATH}")
    print(f"Saved model metrics to: {METRICS_PATH}")
    print(f"Saved weight tuning details to: {WEIGHT_PATH}")


if __name__ == "__main__":
    main()
