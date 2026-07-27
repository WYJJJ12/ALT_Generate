# -*- coding: utf-8 -*-
"""
Annual ALT mapping using final LightGBM and CatBoost models.

The two model predictions are averaged pixel by pixel:
    ALT = (LightGBM prediction + CatBoost prediction) / 2

"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm


warnings.filterwarnings("ignore")


# =========================================================
# 1. Configuration
# =========================================================

BASE_DIR = Path(r"F:\permafrost\ALT_mapping\CALM\20260721\0722")
MODEL_DIR = BASE_DIR / "model_code" / "model_training_outputs" / "final_models_pkl"
OUTPUT_DIR = BASE_DIR / "model_code" / "ALT_mapping_outputs"

START_YEAR = 2000
END_YEAR = 2024
DEFAULT_BLOCK_SIZE = 1024
OUTPUT_NODATA = -9999.0
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

MODEL_FILES = {
    "LightGBM": MODEL_DIR / "LightGBM_ALT_final_model.pkl",
    "CatBoost": MODEL_DIR / "CatBoost_ALT_final_model.pkl",
}

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

CATEGORICAL_VARIABLES = {"Biome"}


# =========================================================
# 2. Arguments and models
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map annual ALT from predictor rasters.")
    parser.add_argument("start_year", nargs="?", type=int, default=START_YEAR)
    parser.add_argument("end_year", nargs="?", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.end_year is None:
        args.end_year = args.start_year
    return args


def load_pickle_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as file:
        payload = pickle.load(file)

    if isinstance(payload, dict) and "model" in payload:
        saved_predictors = payload.get("metadata", {}).get("predictors")
        if saved_predictors is not None and list(saved_predictors) != PREDICTORS:
            raise ValueError(
                f"Predictor order mismatch in {path.name}\n"
                f"Model predictors: {saved_predictors}\n"
                f"Mapping predictors: {PREDICTORS}"
            )
        return payload["model"]
    return payload


def load_models() -> dict[str, object]:
    models = {}
    for model_name, model_path in MODEL_FILES.items():
        models[model_name] = load_pickle_model(model_path)
        print(f"Loaded {model_name}: {model_path}")
    return models


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


def write_prediction_block(dst, values: np.ndarray, valid_mask: np.ndarray, window: Window) -> None:
    block = np.full((int(window.height), int(window.width)), OUTPUT_NODATA, dtype=np.float32)
    block[valid_mask] = values.astype(np.float32)
    dst.write(block, 1, window=window)


# =========================================================
# 4. Mapping
# =========================================================

def map_one_year(
    year: int,
    models: dict[str, object],
    block_size: int,
    overwrite: bool,
) -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ALT_{year}.tif"
    tmp_path = OUTPUT_DIR / f"ALT_{year}.tmp.tif"

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Existing output: {output_path}")
        return True
    if tmp_path.exists():
        tmp_path.unlink()

    raster_items = get_year_raster_items(year)
    check_files(raster_items)
    sources, raw_sources, ref_grid = open_aligned_sources(raster_items)
    profile = create_output_profile(ref_grid["profile"])
    dst = None

    try:
        width = ref_grid["width"]
        height = ref_grid["height"]
        dst = rasterio.open(tmp_path, "w", **profile)

        block_positions = [
            (row, col)
            for row in range(0, height, block_size)
            for col in range(0, width, block_size)
        ]

        for row, col in tqdm(block_positions, desc=f"Mapping ALT {year}", unit="block"):
            window = Window(
                col_off=col,
                row_off=row,
                width=min(block_size, width - col),
                height=min(block_size, height - row),
            )
            env_stack, valid_mask = read_block_stack(sources, window)

            if not np.any(valid_mask):
                empty = np.full((int(window.height), int(window.width)), OUTPUT_NODATA, dtype=np.float32)
                dst.write(empty, 1, window=window)
                continue

            X_valid = env_stack[valid_mask]
            pred_lightgbm = np.asarray(models["LightGBM"].predict(X_valid), dtype=np.float32)
            pred_catboost = np.asarray(models["CatBoost"].predict(X_valid), dtype=np.float32)
            ensemble_pred = ((pred_lightgbm + pred_catboost) / 2.0).astype(np.float32)

            if CLIP_NEGATIVE_ALT:
                ensemble_pred = np.maximum(ensemble_pred, 0.0)

            write_prediction_block(dst, ensemble_pred, valid_mask, window)
            del env_stack, X_valid, pred_lightgbm, pred_catboost, ensemble_pred
            gc.collect()

        dst.close()
        dst = None
        if output_path.exists() and overwrite:
            output_path.unlink()
        tmp_path.replace(output_path)
        print(f"Saved: {output_path}")
        return True

    except Exception:
        if dst is not None:
            dst.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    finally:
        for src in sources:
            if isinstance(src, WarpedVRT):
                src.close()
        for src in raw_sources:
            src.close()
        gc.collect()


def main() -> None:
    args = parse_args()
    print(f"Years: {args.start_year}-{args.end_year}")
    print(f"Predictor order: {PREDICTORS}")
    print(f"Output directory: {OUTPUT_DIR}")

    models = load_models()
    completed = []
    failed = []
    for year in range(args.start_year, args.end_year + 1):
        try:
            if map_one_year(year, models, args.block_size, args.overwrite):
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
