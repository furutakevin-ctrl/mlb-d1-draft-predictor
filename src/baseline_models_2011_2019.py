"""Run the same baseline logistic-regression model set on 2011-2019 data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2011_2019.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "baseline_models_2011_2019.txt"
PREDICTION_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_cv_predictions_2011_2019.csv"

df = pd.read_csv(DATA_PATH)
target = "made_mlb_ever"
needed = {
    target, "draft_year", "overall_pick", "position", "college_PA", "college_OPS",
    "college_BB_rate", "college_K_rate", "college_HR_rate", "team_SOS", "age_at_draft",
}
missing = sorted(needed - set(df.columns))
if missing:
    raise ValueError(f"Combined modeling table is missing columns: {missing}")

for column in [target, "draft_year", "overall_pick"]:
    df[column] = pd.to_numeric(df[column], errors="raise")
if not df[target].isin([0, 1]).all():
    raise ValueError(f"{target} must contain only 0 and 1.")
if df[["draft_year", "overall_pick"]].duplicated().any():
    raise ValueError("Duplicate draft-year/pick rows found in combined modeling table.")

df["position_family"] = df["position"].astype("string").str.strip()
df.loc[df["position_family"].isin(["CF", "LF", "RF", "OF"]), "position_family"] = "OF"
df.loc[df["position_family"].isin(["1B", "2B", "3B", "SS", "IF"]), "position_family"] = "IF"

y = df[target].astype(int)
cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_pipeline(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )


college_features = [
    "college_PA", "college_OPS", "college_BB_rate", "college_K_rate", "college_HR_rate",
]
model_specs = {
    "draft_pick_only": {"numeric": ["overall_pick"], "categorical": []},
    "draft_pick_plus_position": {"numeric": ["overall_pick"], "categorical": ["position_family"]},
    "draft_pick_plus_college": {"numeric": ["overall_pick", *college_features], "categorical": ["position_family"]},
    "draft_pick_plus_college_sos": {"numeric": ["overall_pick", *college_features, "team_SOS"], "categorical": ["position_family"]},
    "draft_pick_plus_college_age": {"numeric": ["overall_pick", *college_features, "age_at_draft"], "categorical": ["position_family"]},
    "draft_pick_plus_college_age_sos": {"numeric": ["overall_pick", *college_features, "age_at_draft", "team_SOS"], "categorical": ["position_family"]},
    "college_stats_only": {"numeric": college_features, "categorical": []},
    "college_stats_plus_sos": {"numeric": [*college_features, "team_SOS"], "categorical": []},
    "college_stats_plus_age": {"numeric": [*college_features, "age_at_draft"], "categorical": []},
    "college_stats_plus_age_sos": {"numeric": [*college_features, "age_at_draft", "team_SOS"], "categorical": []},
}

predictions = df[["player_name", "draft_year", "overall_pick", "position_family", target]].copy()
results = []
baseline_probability = y.mean()
baseline_brier = brier_score_loss(y, np.repeat(baseline_probability, len(y)))

for model_name, feature_spec in model_specs.items():
    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]
    feature_columns = numeric_features + categorical_features
    pipeline = build_pipeline(numeric_features, categorical_features)
    probabilities = cross_val_predict(
        pipeline, df[feature_columns], y, cv=cross_validator, method="predict_proba"
    )[:, 1]
    probabilities = np.clip(probabilities, 0.001, 0.999)
    predictions[model_name] = probabilities
    results.append(
        {
            "model": model_name,
            "roc_auc": roc_auc_score(y, probabilities),
            "average_precision": average_precision_score(y, probabilities),
            "brier_score": brier_score_loss(y, probabilities),
            "log_loss": log_loss(y, probabilities),
        }
    )

results_df = pd.DataFrame(results)
year_summary = (
    df.groupby("draft_year")
    .agg(players=("player_name", "size"), positive_outcomes=(target, "sum"), mlb_rate=(target, "mean"))
    .reset_index()
)
report_lines = [
    "2011-2019 BASELINE MODEL COMPARISON",
    "=" * 38,
    "",
    f"Players: {len(df)}",
    f"Positive outcomes: {int(y.sum())}",
    f"Overall MLB rate: {baseline_probability:.1%}",
    "",
    "Rows and outcomes by draft year:",
    year_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    "",
    "Constant-probability baseline:",
    f"Brier score: {baseline_brier:.4f}",
    "",
    "Stratified 5-fold cross-validated results:",
    results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
    "",
    "Metric meanings:",
    "ROC-AUC: ranking quality; higher is better.",
    "Average precision: positive-class ranking; higher is better.",
    "Brier score: probability accuracy; lower is better.",
    "Log loss: probability penalty; lower is better.",
    "",
    "Validation note:",
    "Random folds contain players from all draft years; these metrics describe pooled cross-validation, not a future-year holdout.",
    "The made_mlb_ever outcome has different follow-up lengths by draft year; review the year-level rates above.",
]
report = "\n".join(report_lines)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(report, encoding="utf-8")
predictions.to_csv(PREDICTION_PATH, index=False)
print(report)
print(f"\nSaved report to: {REPORT_PATH}")
print(f"Saved predictions to: {PREDICTION_PATH}")
