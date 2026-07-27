# ALT Modelling, Mapping, and Uncertainty Estimation

This folder contains the Python scripts used to train active layer thickness
(ALT) models, generate annual ALT maps, and estimate spatial prediction
uncertainty.

The scripts assume that the modelling dataset has already been prepared. They
do not perform outlier removal, residual filtering, or manual data-cleaning
steps.

## Scripts

### 1. Model Training and Evaluation

`01_model_training_loso_and_save_models.py`

This script:

- reads the final modelling dataset;
- evaluates five candidate models using leave-one-site-out cross-validation;
- evaluates the LightGBM-CatBoost ensemble;
- trains final LightGBM and CatBoost models using all available records;
- saves the final models as `.pkl` files.

Main outputs:

- `model_training_outputs/LOSO_model_evaluation_results.xlsx`
- `model_training_outputs/LOSO_model_predictions.csv`
- `model_training_outputs/final_models_pkl/LightGBM_ALT_final_model.pkl`
- `model_training_outputs/final_models_pkl/CatBoost_ALT_final_model.pkl`

Run:

```bash
python 01_model_training_loso_and_save_models.py
```

### 2. Annual ALT Mapping

`02_map_ALT_annual.py`

This script:

- loads the final LightGBM and CatBoost `.pkl` models;
- reads annual predictor rasters;
- predicts ALT using both models;
- averages the two predictions to produce the ensemble ALT map.

Main output:

- `ALT_mapping_outputs/ALT_YYYY.tif`

Run all years:

```bash
python 02_map_ALT_annual.py 2000 2024
```

Run one year:

```bash
python 02_map_ALT_annual.py 2000
```

Overwrite existing outputs:

```bash
python 02_map_ALT_annual.py 2000 2024 --overwrite
```

### 3. Site-Level Subsampling Uncertainty

`03_site_subsampling_uncertainty.py`

This script estimates prediction uncertainty using repeated site-level
subsampling.

Workflow:

- randomly sample 80% of training sites without replacement;
- include all records from selected sites in model training;
- train LightGBM and CatBoost using the same sampled sites;
- average the two model predictions to obtain one ensemble prediction;
- repeat the procedure many times;
- calculate pixel-wise uncertainty statistics.

Main outputs:

- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_mean_YYYY.tif`
- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_sd_YYYY.tif`
- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_p025_YYYY.tif`
- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_p975_YYYY.tif`
- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_pi_width_YYYY.tif`
- `uncertainty_outputs/site_subsample_XXiter_80pct/ALT_subsample_relative_uncertainty_YYYY.tif`

Relative uncertainty is calculated as:

```text
(P97.5 - P2.5) / mean_prediction * 100
```

Run 50 iterations for one year:

```bash
python 03_site_subsampling_uncertainty.py 2000 2000 --n-iter 50 --sample-frac 0.8
```

Run 100 iterations for all years:

```bash
python 03_site_subsampling_uncertainty.py 2000 2024 --n-iter 100 --sample-frac 0.8
```

## Input Data

The modelling table is configured in the scripts as:

```text
F:\permafrost\202503_ALT_mapping\CALM\20260721\0722\ALT_variables.xlsx
```

Sheet name:

```text
ALT0722
```

Required columns:

```text
Site Name, LAT, LONG, ALT,
SR, MAT, MAP, SDE, TI, NDVI, Elev, Slope, Biome, BD, CF, Silt
```

The predictor order must remain:

```text
SR, MAT, MAP, SDE, TI, NDVI, Elev, Slope, Biome, BD, CF, Silt
```

The same predictor order is used for model training, annual mapping, and
uncertainty estimation.

## Predictor Rasters

The mapping and uncertainty scripts use yearly rasters for:

- solar radiation (`SR`);
- mean annual temperature (`MAT`);
- annual precipitation (`MAP`);
- snow depth (`SDE`);
- thawing index (`TI`);
- MODIS NDVI (`NDVI`).

They use static rasters for:

- elevation (`Elev`);
- slope (`Slope`);
- biome class (`Biome`);
- bulk density (`BD`);
- coarse fragments (`CF`);
- silt content (`Silt`).

Raster paths are defined in the `DYNAMIC_RASTERS` and `STATIC_RASTERS`
dictionaries at the top of the mapping and uncertainty scripts.

## Software Dependencies

The scripts require Python with the following packages:

```text
numpy
pandas
scikit-learn
lightgbm
xgboost
catboost
rasterio
tqdm
openpyxl
```

## Notes

- Leave-one-site-out validation groups records by normalized site name and
  rounded coordinates.
- The final ALT map is the arithmetic mean of LightGBM and CatBoost
  predictions.
- Negative predicted ALT values are clipped to zero before writing maps.
- Continuous rasters are bilinearly aligned when needed; the categorical
  `Biome` raster is aligned using nearest-neighbor resampling.
- Output rasters are written as compressed Float32 GeoTIFF files with
  `-9999` as NoData.
