# JPX Tokyo Stock Exchange Prediction

Stock return ranking with a two-stage stacking model (Ridge + LightGBM).

## Environment

- Python 3.12+
- Tested on Windows with CUDA-capable GPU (GPU not required)

```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download the competition dataset from [Kaggle](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/data)
2. Extract the archive and place the `jpx-tokyo-stock-exchange-prediction/` folder **inside this `source_code/` directory** (same level as this README)

The expected structure:

```
source_code/
├── jpx-tokyo-stock-exchange-prediction/   <-- place downloaded data here
│   ├── train_files/
│   │   ├── stock_prices.csv
│   │   ├── secondary_stock_prices.csv
│   │   ├── financials.csv
│   │   └── ...
│   ├── supplemental_files/
│   │   ├── stock_prices.csv
│   │   ├── secondary_stock_prices.csv
│   │   └── ...
│   └── stock_list.csv
├── src/
├── output/
├── run_pipeline.py
├── evaluate.py
└── README.md
```

> **Note**: The dataset is ~1.2 GB and is excluded from this repository (see `.gitignore`). If you prefer to keep the data in a different location, set the environment variable before running:
>
> ```bash
> # Windows
> set JPX_DATA_DIR=C:\path\to\jpx-tokyo-stock-exchange-prediction
>
> # Linux / macOS
> export JPX_DATA_DIR=/path/to/jpx-tokyo-stock-exchange-prediction
> ```

## Quick Start

### Option A: Evaluate Pre-trained Models

If you want to skip training and directly evaluate the provided models:

```bash
python evaluate.py
```

This loads the pre-trained Ridge + LightGBM stacking models from `output/models/` and reports Sharpe ratio and Spearman correlation across multiple time periods.

### Option B: Train from Scratch

```bash
python run_pipeline.py
```

This runs the full pipeline:
1. Load all price data (primary + supplemental + secondary stocks)
2. Preprocess with z-score normalization
3. Build 18 features (9 base + 9 cross-sectional rank)
4. Train stacking model (Ridge at level-0, LightGBM on residuals at level-1)
5. Save trained models to `output/models/`

Then evaluate:

```bash
python evaluate.py
```

## Project Structure

```
src/
├── config.py                 # Paths, hyperparameters, feature definitions
├── data/
│   ├── loader.py             # Load primary + secondary stock prices
│   └── preprocessor.py       # Fill missing values, z-score normalization
├── features/
│   ├── build_features.py     # Feature orchestration and caching
│   ├── price_features.py     # Returns, volatility, intraday range, lagged target
│   └── cross_section.py      # Cross-sectional percentile rank features
├── models/
│   └── train_lgb.py          # Stacking model: Ridge (L0) + LightGBM (L1)
└── evaluation/
    └── metrics.py            # Spread return Sharpe ratio, Spearman correlation

output/models/                 # Pre-trained models
├── ridge_stacking.joblib      # Ridge regression model
├── gbm_stacking.txt           # LightGBM model
└── cs_features.txt            # Feature list for inference

run_pipeline.py                # Main training script
evaluate.py                    # Model evaluation with per-period breakdown
notebooks/
└── demo.ipynb                 # Interactive demo with visualizations
```

## Method

### Problem

Given daily stock prices for ~2,000 Japanese stocks, predict the next-day relative return ranking. Evaluation uses Spread Return Sharpe Ratio: long the top 200 and short the bottom 200 stocks by predicted rank, with linearly decaying weights.

### Architecture: Two-Stage Stacking

**Level 0 — Ridge Regression** on z-scored raw price features:
- Per-stock z-score normalization of Open, High, Low, Close, Volume makes prices comparable across stocks with different price levels
- 9 base features: `Open_z`, `High_z`, `Low_z`, `Close_z`, `Volume_z`, `Daily_Range_z`, `Mean_z`, `SupervisionFlag`, `ExpectedDividend`
- Captures linear price-level signal

**Level 1 — LightGBM** on cross-sectional rank features, trained on Ridge residuals:
- 9 rank features: percentile rank of `Return_1d`, `Return_5d`, `Return_10d`, `Return_20d`, `Volatility_20d`, `TargetLag_1d`, `Close_z`, `Volume_z`, `IntradayRange`
- Trained to predict `Target - Ridge_prediction` (residuals)
- Captures non-linear cross-sectional patterns the linear model misses

**Final prediction** = Ridge prediction + LightGBM residual prediction

### Key Design Decisions

1. **Z-score normalization**: Raw prices are incomparable across stocks. Per-stock z-scoring makes the price level meaningful as a feature (high z-score = stock is expensive relative to its own history).
2. **Use all data sources**: Including secondary stocks doubles training data and significantly improves test performance.
3. **Feature separation**: Base features (for Ridge) and rank features (for LightGBM) serve different purposes — one captures level, the other captures relative positioning.
4. **Residual stacking**: Rather than ensembling predictions, we train LightGBM on what Ridge cannot explain, avoiding redundancy.

### Results

| Model | Valid Sharpe | Test Sharpe |
|-------|-------------|-------------|
| Ridge only | 0.62 | 0.46 |
| Ridge + LightGBM (stacking) | 1.41 | 0.62 |

Per-year test breakdown:

| Period | Sharpe |
|--------|--------|
| 2021 H2 | 0.93 |
| 2022 | 0.43 |

## Notes

- Feature caching is enabled by default (`output/features/features_v6.parquet`). Delete this file to force recomputation.
- Training takes ~5-10 minutes on a machine with 8GB+ RAM. GPU is not used.
- The dataset contains ~5.3 million rows across 4,416 stocks and 1,337 trading days.
