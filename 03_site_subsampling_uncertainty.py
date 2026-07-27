# -*- coding: utf-8 -*-
"""
Site-level repeated subsampling uncertainty mapping for ALT.

Workflow
--------
1. Randomly sample a fixed fraction of training sites without replacement.
2. Use all observations from selected sites to train LightGBM and CatBoost.
3. Average LightGBM and CatBoost predictions to obtain one ensemble map.
4. Repeat the procedure many times.
5. Calculate pixel-wise mean, SD, P2.5, P97.5, prediction-interval width,
   and relative uncertainty.

Relative uncertainty is calculated as:
    (P97.5 - P2.5) / mean_prediction * 100

"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import rasterio
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm


warnings.filterwarnings("ignore")


# =========================================================
# 1. Configuration
# =========================================================

BASE_DIR = Path(r"F:\permafrost\ALT_mapping\CALM\20260721\0722")
INPUT_FILE = BASE_DIR / "ALT_variables.xlsx"
SHEET_NAME = "ALT0722"
OUTPUT_ROOT = BASE_DIR / "model_code" / "uncertainty_outputs"

START_YEAR = 2000
END_YEAR = 2024
DEFAULT_N_ITER = 100
DEFAULT_SAMPLE_FRAC = 0.80
DEFAULT_BLOCK_SIZE = 512
BASE_RANDOM_SEED = 42
COORD_DECIMALS = 4

TARGET = "ALT"
OUTPUT_NODATA = -9999.0
RELATIVE_EPSILON = 1.0e-6
CLIP_NEGATIVE_ALT = True

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
CATEGORICAL_VARIABLES = {"Biome"}

DYNAMIC_RASTERS = {
    "SR": (
        r"F:\permafrost\Variables\Climate"
        r"\Solar_radiation\annual_radiation_{year}.tif"
    ),
    "MAT": (
        r"F:\permafrost\Variables\Climate"
        r"\temp_year_mean\annual_mean_temp_{year}.tif"
    ),
    "MAP": (
        r"F:\permafrost\Variables\Climate"
        r"\Pre_year\annual_precip_{year}.tif"
    ),
    "SDE": (
        r"F:\permafrost\Variables\Climate"
        r"\Snow_Depth\sde_{year}_mean.tif"
    ),
    "TI": (
        r"F:\permafrost\Variables\Climate"
        r"\TI\TI_{year}.tif"
    ),
    "NDVI": (
        r"F:\permafrost\Variables\Organism\NDVI"
        r"\MODIS_NDVI_1000M_EPSG3995_2000_2024"
        r"\MODIS_NDVI_1000M_EPSG3995_{year}.tif"
    ),
}

STATIC_RASTERS = {
    "Elev": r"F:\permafrost\Variables\Topography\DEM.tif",
    "Slope": r"F:\permafrost\Variables\Topography\new\Slope.tif",
    "Biome": r"F:\permafrost\Variables\Organism\biome.tif",
    "BD": r"F:\permafrost\Variables\Soil\bdod.tif",
    "CF": r"F:\permafrost\Variables\Soil\cfvo.tif",
    "Silt": r"F:\permafrost\Variables\Soil\silt.tif",
}


# =========================================================
# 2. Arguments and training data
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Site-level subsampling uncertainty mapping for ALT.")
    parser.add_argument("start_year", nargs="?", type=int, default=START_YEAR)
    parser.add_argument("end_year", nargs="?", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--sample-frac", type=float, default=DEFAULT_SAMPLE_FRAC)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--seed", type=int, default=BASE_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.end_year is None:
        args.end_year = args.start_year
    if args.n_iter < 2:
        raise ValueError("--n-iter must be at least 2.")
    if not 0.0 < args.sample_frac <= 1.0:
        raise ValueError("--sample-frac must be in (0, 1].")
    return args


def normalize_site_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def load_training_data() -> pd.DataFrame:
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
    print("Training data for uncertainty modelling")
    print(f"Workbook: {INPUT_FILE}")
    print(f"Sheet: {SHEET_NAME}")
    print(f"Input records: {n_before}")
    print(f"Records used: {len(df)}")
    print(f"Site groups: {df['Site_ID'].nunique()}")
    print(f"Predictors ({len(PREDICTORS)}): {', '.join(PREDICTORS)}")
    print("=" * 70)
    return df.reset_index(drop=True)


def build_model_factories(seed: int) -> dict[str, Callable[[], object]]:
    return {
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


def train_subsample_models(
    df: pd.DataFrame,
    n_iter: int,
    sample_frac: float,
    seed: int,
) -> list[tuple[object, object]]:
    site_ids = np.array(sorted(df["Site_ID"].unique()))
    n_sites = len(site_ids)
    n_sample = max(1, int(round(n_sites * sample_frac)))
    rng = np.random.default_rng(seed)
    model_pairs = []

    print(f"Training {n_iter} subsampled LightGBM-CatBoost ensemble models ...")
    print(f"Sites per iteration: {n_sample}/{n_sites} ({sample_frac:.0%})")

    for i in range(n_iter):
        iter_seed = seed + i
        selected_sites = rng.choice(site_ids, size=n_sample, replace=False)
        train_df = df.loc[df["Site_ID"].isin(selected_sites)]

        X = train_df[PREDICTORS].to_numpy(dtype=float)
        y = train_df[TARGET].to_numpy(dtype=float)
        factories = build_model_factories(iter_seed)

        lightgbm_model = factories["LightGBM"]()
        catboost_model = factories["CatBoost"]()
        lightgbm_model.fit(X, y)
        catboost_model.fit(X, y)
        model_pairs.append((lightgbm_model, catboost_model))

        print(
            f"[{i + 1:03d}/{n_iter:03d}] "
            f"trained with {len(train_df)} records from {len(selected_sites)} sites"
        )

    return model_pairs


# =========================================================
# 3. Raster helpers
# =========================================================

def get_year_raster_items(year: int) -> list[tuple[str, Path]]:
    paths = {}
    for var_name, pattern in DYNAMIC_RASTERS.items():
        paths[var_name] = Path(pattern.format(year=year))
    for var_name, path in STATIC_RASTERS.items():
        paths[var_name] = Path(path)
    return [(var_name, paths[var_name]) for var_name in PREDICTORS]


def check_files(raster_items: list[tuple[str, Path]]) -> None:
    missing = [f"{var}: {path}" for var, path in raster_items if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing raster files:\n" + "\n".join(missing))


def get_reference_grid(reference_path: Path) -> dict:
    with rasterio.open(reference_path) as ref:
        return {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "profile": ref.profile.copy(),
        }


def open_aligned_sources(raster_items: list[tuple[str, Path]]):
    ref_grid = get_reference_grid(raster_items[0][1])
    raw_sources = []
    aligned_sources = []

    for var_name, path in raster_items:
        src = rasterio.open(path)
        raw_sources.append(src)
        is_aligned = (
            src.crs == ref_grid["crs"]
            and src.width == ref_grid["width"]
            and src.height == ref_grid["height"]
            and src.transform.almost_equals(ref_grid["transform"])
        )
        if is_aligned:
            aligned_sources.append(src)
        else:
            resampling = Resampling.nearest if var_name in CATEGORICAL_VARIABLES else Resampling.bilinear
            print(f"[WARN] Aligning {var_name} by WarpedVRT: {path}")
            aligned_sources.append(
                WarpedVRT(
                    src,
                    crs=ref_grid["crs"],
                    transform=ref_grid["transform"],
                    width=ref_grid["width"],
                    height=ref_grid["height"],
                    resampling=resampling,
                    nodata=src.nodata,
                )
            )

    return aligned_sources, raw_sources, ref_grid


def create_output_profile(reference_profile: dict) -> dict:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=OUTPUT_NODATA,
        compress="lzw",
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return profile


def read_block_stack(open_sources, window: Window) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for src in open_sources:
        masked = src.read(1, window=window, masked=True).astype(np.float32)
        arr = masked.filled(np.nan).astype(np.float32)
        arr[~np.isfinite(arr)] = np.nan
        if src.nodata is not None and np.isfinite(src.nodata):
            arr[arr == src.nodata] = np.nan
        arrays.append(arr)

    env_stack = np.stack(arrays, axis=-1)
    valid_mask = np.isfinite(env_stack).all(axis=-1)
    return env_stack, valid_mask


def make_empty_block(window: Window) -> np.ndarray:
    return np.full((int(window.height), int(window.width)), OUTPUT_NODATA, dtype=np.float32)


def fill_output_block(values: np.ndarray, valid_mask: np.ndarray, window: Window) -> np.ndarray:
    block = make_empty_block(window)
    block[valid_mask] = values.astype(np.float32)
    return block


def fill_output_block_allow_nan(
    values: np.ndarray,
    valid_mask: np.ndarray,
    window: Window,
) -> np.ndarray:
    block = make_empty_block(window)
    clean_values = values.astype(np.float32)
    clean_values[~np.isfinite(clean_values)] = OUTPUT_NODATA
    block[valid_mask] = clean_values
    return block


# =========================================================
# 4. Uncertainty mapping
# =========================================================

def get_output_paths(output_dir: Path, year: int) -> dict[str, Path]:
    return {
        "mean": output_dir / f"ALT_mean_{year}.tif",
        "sd": output_dir / f"ALT_sd_{year}.tif",
        "p025": output_dir / f"ALT_p025_{year}.tif",
        "p975": output_dir / f"ALT_p975_{year}.tif",
        "pi_width": output_dir / f"ALT_pi_width_{year}.tif",
        "relative_uncertainty": output_dir / f"ALT_relative_uncertainty_{year}.tif",
    }


def all_outputs_exist(output_paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in output_paths.values())


def remove_tmp_files(output_paths: dict[str, Path]) -> None:
    for path in output_paths.values():
        tmp_path = path.with_suffix(".tmp.tif")
        if tmp_path.exists():
            tmp_path.unlink()


def map_uncertainty_one_year(
    year: int,
    model_pairs: list[tuple[object, object]],
    output_dir: Path,
    block_size: int,
    overwrite: bool,
) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = get_output_paths(output_dir, year)

    if all_outputs_exist(output_paths) and not overwrite:
        print(f"[SKIP] Existing uncertainty outputs for {year}")
        return True
    remove_tmp_files(output_paths)

    raster_items = get_year_raster_items(year)
    check_files(raster_items)
    sources, raw_sources, ref_grid = open_aligned_sources(raster_items)
    profile = create_output_profile(ref_grid["profile"])
    datasets = {}

    try:
        for name, path in output_paths.items():
            datasets[name] = rasterio.open(path.with_suffix(".tmp.tif"), "w", **profile)

        width = ref_grid["width"]
        height = ref_grid["height"]
        block_positions = [
            (row, col)
            for row in range(0, height, block_size)
            for col in range(0, width, block_size)
        ]

        for row, col in tqdm(block_positions, desc=f"Uncertainty ALT {year}", unit="block"):
            window = Window(
                col_off=col,
                row_off=row,
                width=min(block_size, width - col),
                height=min(block_size, height - row),
            )
            env_stack, valid_mask = read_block_stack(sources, window)

            if not np.any(valid_mask):
                empty = make_empty_block(window)
                for dst in datasets.values():
                    dst.write(empty, 1, window=window)
                continue

            X_valid = env_stack[valid_mask]
            n_valid = X_valid.shape[0]
            predictions = np.empty((len(model_pairs), n_valid), dtype=np.float32)

            for i, (lightgbm_model, catboost_model) in enumerate(model_pairs):
                pred_lightgbm = np.asarray(lightgbm_model.predict(X_valid), dtype=np.float32)
                pred_catboost = np.asarray(catboost_model.predict(X_valid), dtype=np.float32)
                ensemble_pred = (pred_lightgbm + pred_catboost) / 2.0
                if CLIP_NEGATIVE_ALT:
                    ensemble_pred = np.maximum(ensemble_pred, 0.0)
                predictions[i, :] = ensemble_pred

            mean_pred = np.mean(predictions, axis=0, dtype=np.float64).astype(np.float32)
            sd_pred = np.std(predictions, axis=0, ddof=1).astype(np.float32)
            p025_pred = np.percentile(predictions, 2.5, axis=0).astype(np.float32)
            p975_pred = np.percentile(predictions, 97.5, axis=0).astype(np.float32)
            pi_width = (p975_pred - p025_pred).astype(np.float32)
            relative_uncertainty = np.full_like(mean_pred, np.nan, dtype=np.float32)
            positive_mean = mean_pred > RELATIVE_EPSILON
            relative_uncertainty[positive_mean] = (
                pi_width[positive_mean] / mean_pred[positive_mean] * 100.0
            ).astype(np.float32)

            datasets["mean"].write(fill_output_block(mean_pred, valid_mask, window), 1, window=window)
            datasets["sd"].write(fill_output_block(sd_pred, valid_mask, window), 1, window=window)
            datasets["p025"].write(fill_output_block(p025_pred, valid_mask, window), 1, window=window)
            datasets["p975"].write(fill_output_block(p975_pred, valid_mask, window), 1, window=window)
            datasets["pi_width"].write(fill_output_block(pi_width, valid_mask, window), 1, window=window)
            datasets["relative_uncertainty"].write(
                fill_output_block_allow_nan(relative_uncertainty, valid_mask, window),
                1,
                window=window,
            )

            del env_stack, X_valid, predictions
            del mean_pred, sd_pred, p025_pred, p975_pred, pi_width, relative_uncertainty
            gc.collect()

        for dst in datasets.values():
            dst.close()
        datasets.clear()

        for path in output_paths.values():
            tmp_path = path.with_suffix(".tmp.tif")
            if path.exists() and overwrite:
                path.unlink()
            tmp_path.replace(path)
            print(f"Saved: {path}")

        return True

    except Exception:
        for dst in datasets.values():
            dst.close()
        remove_tmp_files(output_paths)
        raise

    finally:
        for src in sources:
            if isinstance(src, WarpedVRT):
                src.close()
        for src in raw_sources:
            src.close()
        gc.collect()


def write_run_metadata(output_dir: Path, df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "site-level repeated subsampling without replacement",
        "ensemble": "mean of LightGBM and CatBoost predictions",
        "n_iter": args.n_iter,
        "sample_frac": args.sample_frac,
        "random_seed": args.seed,
        "years": f"{args.start_year}-{args.end_year}",
        "target": TARGET,
        "predictors": ", ".join(PREDICTORS),
        "input_file": str(INPUT_FILE),
        "sheet_name": SHEET_NAME,
        "n_records": len(df),
        "n_sites": int(df["Site_ID"].nunique()),
        "relative_uncertainty": "(P97.5 - P2.5) / mean_prediction * 100",
        "nodata": OUTPUT_NODATA,
    }
    metadata_path = output_dir / "uncertainty_metadata.txt"
    metadata_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        encoding="utf-8",
    )
    print(f"Saved: {metadata_path}")


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / f"site_subsample_{args.n_iter}iter_{int(args.sample_frac * 100):02d}pct"

    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Iterations: {args.n_iter}")
    print(f"Site sample fraction: {args.sample_frac}")
    print(f"Block size: {args.block_size}")
    print(f"Output directory: {output_dir}")

    df = load_training_data()
    write_run_metadata(output_dir, df, args)
    model_pairs = train_subsample_models(df, args.n_iter, args.sample_frac, args.seed)

    completed = []
    failed = []
    for year in range(args.start_year, args.end_year + 1):
        try:
            if map_uncertainty_one_year(
                year=year,
                model_pairs=model_pairs,
                output_dir=output_dir,
                block_size=args.block_size,
                overwrite=args.overwrite,
            ):
                completed.append(year)
        except Exception as error:
            print(f"[FAILED] {year}: {error}")
            failed.append(year)

    print("=" * 70)
    print(f"Completed years: {completed}")
    print(f"Failed years: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
