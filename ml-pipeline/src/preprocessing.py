import os
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .utils import get_logger, save_joblib, save_json

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(dataset_path: str, excel_sheet: Optional[str]=None) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    _, ext = os.path.splitext(dataset_path.lower())
    if ext in [".csv", ".txt"]:
        return pd.read_csv(dataset_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(dataset_path, sheet_name=excel_sheet)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Use dense output to ensure downstream models (e.g., trees, lbfgs solver) can fit
        # scikit-learn >=1.2 uses 'sparse_output' instead of deprecated 'sparse'
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor

def preprocess(cfg_path: str) -> Tuple[str, str, str]:
    logger = get_logger("preprocessing")
    cfg = load_config(cfg_path)

    df = load_dataset(cfg["dataset_path"], cfg.get("excel_sheet"))
    logger.info(f"Loaded dataset with shape {df.shape} from {cfg['dataset_path']}")

    # Drop ID columns if provided
    id_cols = cfg.get("id_columns") or []
    for col in id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    target_col = cfg["target_col"]
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataset columns: {df.columns.tolist()}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Log basic schema info
    num_cols = X.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()
    logger.info(f"Identified columns -> numeric: {len(num_cols)}, categorical: {len(cat_cols)}")

    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    logger.info(f"Transformed feature matrix shape: {getattr(X_processed, 'shape', None)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y if y.nunique() > 1 else None
    )

    # Save artifacts
    transformer_path = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
    Xy_train_path = os.path.join(ARTIFACTS_DIR, "train_data.joblib")
    Xy_test_path  = os.path.join(ARTIFACTS_DIR, "test_data.joblib")

    save_joblib(transformer_path, preprocessor)
    save_joblib(Xy_train_path, {"X": X_train, "y": y_train.values})
    save_joblib(Xy_test_path,  {"X": X_test,  "y": y_test.values})
    logger.info(f"Saved artifacts: preprocessor -> {transformer_path}, train -> {Xy_train_path}, test -> {Xy_test_path}")

    meta = {
        "n_samples": int(df.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_transformed": int(X_processed.shape[1]),
        "target_name": target_col,
    }
    save_json(os.path.join(ARTIFACTS_DIR, "preprocess_meta.json"), meta)
    logger.info(f"Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}")
    return transformer_path, Xy_train_path, Xy_test_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))
    args = ap.parse_args()
    preprocess(args.config)
