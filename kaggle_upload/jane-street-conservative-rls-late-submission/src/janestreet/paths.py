"""Project data paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KAGGLE_DOWNLOAD_DIR = RAW_DATA_DIR / "kaggle"
COMPETITION_DATA_DIR = RAW_DATA_DIR / "jane-street-real-time-market-data-forecasting"
TRAIN_PARQUET_DIR = COMPETITION_DATA_DIR / "train.parquet"
TRAIN_WITH_RESPONDER_LAGS_PARQUET = PROCESSED_DATA_DIR / "train_with_responder_lags.parquet"
DAILY_RESPONDER_LAGS_LAST_PARQUET = PROCESSED_DATA_DIR / "daily_responder_lags_last.parquet"
TEST_PARQUET_DIR = COMPETITION_DATA_DIR / "test.parquet"
LAGS_PARQUET_DIR = COMPETITION_DATA_DIR / "lags.parquet"
FEATURES_CSV = COMPETITION_DATA_DIR / "features.csv"
