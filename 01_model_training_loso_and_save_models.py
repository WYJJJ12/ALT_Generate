# -*- coding: utf-8 -*-
"""
Model training and evaluation for active layer thickness (ALT).

Workflow
--------
1. Read the final modelling dataset.
2. Evaluate five candidate models using leave-one-site-out cross-validation.
3. Evaluate a LightGBM-CatBoost ensemble by averaging their predictions.
4. Train final LightGBM and CatBoost models using all available records.
5. Save the final models as .pkl files.

Required columns
----------------
Site Name, LAT, LONG, ALT,
SR, MAT, MAP, SDE, TI, NDVI, Elev, Slope, Biome, BD, CF, Silt
"""

from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from xgboost import XGBRegressor


# =========================================================
# 1. Configuration
# =========================================================

BASE_DIR = Path(r"F:\permafrost\ALT_mapping\CALM\20260721\0722")
INPUT_FILE = BASE_DIR / "ALT_variables.xlsx"
SHEET_NAME = "ALT0722"

OUTPUT_DIR = BASE_DIR / "model_code" / "model_training_outputs"
MODEL_DIR = OUTPUT_DIR / "final_models_pkl"

RANDOM_SEED = 42
COORD_DECIMALS = 4

TARGET = "ALT"
PREDICTORS = [
    "SR",
    "MAT",
    "MAP",
    "SDE",
    "TI",
    "NDVI",
    "Elev",
    "Slope",
    "Biome",
    "BD",
    "CF",
    "Silt",
]
REQUIRED_COLUMNS = ["Site Name", "LAT", "LONG", TARGET] + PREDICTORS
FINAL_MODEL_NAMES = ["LightGBM", "CatBoost"]


# =========================================================
# 2. Data and metrics
# =========================================================

def normalize_site_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def load_prepared_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input workbook was not found: {INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    df.columns = [str(col).strip() for col in df.columns]

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    for col in ["LAT", "LONG", TARGET] + PREDICTORS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    finite_mask = np.isfinite(
        df[["LAT", "LONG", TARGET] + PREDICTORS].to_numpy(dtype=float)
    ).all(axis=1)
    df = df.loc[finite_mask].copy()

    df["Site_Name_Normalized"] = df["Site Name"].map(normalize_site_name)
    df["LAT_Rounded"] = df["LAT"].round(COORD_DECIMALS)
    df["LONG_Rounded"] = df["LONG"].round(COORD_DECIMALS)
    df["Site_Key"] = (
        df["Site_Name_Normalized"]
        + "|"
        + df["LAT_Rounded"].map(lambda x: f"{x:.{COORD_DECIMALS}f}")
        + "|"
        + df["LONG_Rounded"].map(lambda x: f"{x:.{COORD_DECIMALS}f}")
    )
    df["Site_ID"] = pd.factorize(df["Site_Key"], sort=True)[0] + 1

    print("=" * 70)
    print("Prepared modelling data")
    print(f"Workbook: {INPUT_FILE}")
    print(f"Sheet: {SHEET_NAME}")
    print(f"Input records: {n_before}")
    print(f"Records used: {len(df)}")
    print(f"Site groups: {df['Site_ID'].nunique()}")
    print(f"Predictors ({len(PREDICTORS)}): {', '.join(PREDICTORS)}")
    print("=" * 70)

    return df.reset_index(drop=True)


def concordance_correlation_coefficient(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = (
        np.var(obs, ddof=1)
        + np.var(pred, ddof=1)
        + (np.mean(obs) - np.mean(pred)) ** 2
    )
    if denominator == 0:
        return np.nan
    return float(2 * np.cov(obs, pred, ddof=1)[0, 1] / denominator)


def pearson_r(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if np.std(obs) == 0 or np.std(pred) == 0:
        return np.nan
    return float(np.corrcoef(obs, pred)[0, 1])


def calculate_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "R2": float(r2_score(obs, pred)),
        "RMSE_cm": float(mean_squared_error(obs, pred, squared=False)),
        "MAE_cm": float(mean_absolute_error(obs, pred)),
        "Bias_cm": float(np.mean(pred - obs)),
        "CCC": concordance_correlation_coefficient(obs, pred),
        "Pearson_r": pearson_r(obs, pred),
    }


# =========================================================
# 3. Model definitions
# =========================================================

def build_model_factories(seed: int = RANDOM_SEED) -> dict[str, Callable[[], object]]:
    return {
        "RF": lambda: RandomForestRegressor(random_state=seed, n_jobs=-1),
        "GBDT": lambda: GradientBoostingRegressor(random_state=seed),
        "XGBoost": lambda: XGBRegressor(
            random_state=seed,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        ),
        "LightGBM": lambda: LGBMRegressor(
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


# =========================================================
# 4. Leave-one-site-out validation
# =========================================================

def run_loso_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[PREDICTORS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    groups = df["Site_ID"].to_numpy()

    model_factories = build_model_factories()
    predictions = {
        model_name: np.full(len(df), np.nan, dtype=float)
        for model_name in model_factories
    }

    splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
    start_time = time.time()
    for train_index, test_index in tqdm(splits, desc="LOSO sites", unit="site"):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        for model_name, model_factory in model_factories.items():
            model = model_factory()
            model.fit(X_train, y_train)
            predictions[model_name][test_index] = np.asarray(model.predict(X_test), dtype=float)

    print(f"LOSO validation completed in {(time.time() - start_time) / 60.0:.2f} minutes.")

    prediction_df = df[["Site_ID", "Site_Key", "Site Name", "LAT", "LONG", TARGET]].copy()
    metric_rows = []
    for model_name, pred in predictions.items():
        if np.isnan(pred).any():
            raise RuntimeError(f"{model_name} has missing LOSO predictions.")
        prediction_df[f"Pred_{model_name}"] = pred
        prediction_df[f"Residual_{model_name}"] = y - pred
        metric_rows.append(
            {
                "Model": model_name,
                **calculate_metrics(y, pred),
                "N_records": len(df),
                "N_sites": int(df["Site_ID"].nunique()),
                "Validation": "Leave-one-site-out cross-validation",
            }
        )

    ensemble_pred = np.mean(
        [predictions["LightGBM"], predictions["CatBoost"]],
        axis=0,
    )
    ensemble_name = "Ensemble_LightGBM_CatBoost"
    prediction_df[f"Pred_{ensemble_name}"] = ensemble_pred
    prediction_df[f"Residual_{ensemble_name}"] = y - ensemble_pred
    metric_rows.append(
        {
            "Model": ensemble_name,
            **calculate_metrics(y, ensemble_pred),
            "N_records": len(df),
            "N_sites": int(df["Site_ID"].nunique()),
            "Validation": "Leave-one-site-out cross-validation",
        }
    )

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE_cm", ascending=True)
    return metrics_df.reset_index(drop=True), prediction_df


def save_loso_outputs(df: pd.DataFrame, metrics_df: pd.DataFrame, prediction_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = OUTPUT_DIR / "LOSO_model_evaluation_results.xlsx"
    csv_path = OUTPUT_DIR / "LOSO_model_predictions.csv"

    summary_df = pd.DataFrame(
        {
            "Item": [
                "Input workbook",
                "Input sheet",
                "Records used",
                "Site groups",
                "Coordinate rounding decimals",
                "Target",
                "Predictors",
                "Validation",
                "Final ensemble",
            ],
            "Value": [
                str(INPUT_FILE),
                SHEET_NAME,
                len(df),
                int(df["Site_ID"].nunique()),
                COORD_DECIMALS,
                TARGET,
                ", ".join(PREDICTORS),
                "Leave-one-site-out cross-validation",
                "Mean of LightGBM and CatBoost predictions",
            ],
        }
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        prediction_df.to_excel(writer, sheet_name="Predictions", index=False)
        summary_df.to_excel(writer, sheet_name="Data_Summary", index=False)

    prediction_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {excel_path}")
    print(f"Saved: {csv_path}")


# =========================================================
# 5. Train final models with all records
# =========================================================

def train_and_save_final_models(df: pd.DataFrame) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X = df[PREDICTORS].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    factories = build_model_factories()

    metadata = {
        "target": TARGET,
        "predictors": PREDICTORS,
        "random_seed": RANDOM_SEED,
        "n_records": len(df),
        "n_sites": int(df["Site_ID"].nunique()),
        "input_file": str(INPUT_FILE),
        "sheet_name": SHEET_NAME,
        "ensemble": "Mean of LightGBM and CatBoost predictions",
    }

    for model_name in FINAL_MODEL_NAMES:
        model = factories[model_name]()
        print(f"Training final {model_name} model with all records ...")
        model.fit(X, y)

        payload = {
            "model_name": model_name,
            "model": model,
            "metadata": metadata,
        }
        output_path = MODEL_DIR / f"{model_name}_ALT_final_model.pkl"
        with output_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved: {output_path}")

    metadata_path = MODEL_DIR / "model_metadata.txt"
    metadata_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        encoding="utf-8",
    )
    print(f"Saved: {metadata_path}")


def main() -> None:
    df = load_prepared_data()
    metrics_df, prediction_df = run_loso_validation(df)
    print("\nLOSO metrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    save_loso_outputs(df, metrics_df, prediction_df)
    train_and_save_final_models(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
