import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .utils import get_logger, load_joblib, save_joblib, save_json

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(),
    "decision_tree": lambda: DecisionTreeClassifier(),
    "random_forest": lambda: RandomForestClassifier(),
}

def train(cfg_path: str) -> str:
    logger = get_logger("training")
    cfg = load_config(cfg_path)

    train_data = load_joblib(os.path.join(ARTIFACTS_DIR, "train_data.joblib"))
    X_train, y_train = train_data["X"], train_data["y"]
    logger.info(f"Loaded training data. X_train={getattr(X_train, 'shape', None)}, y_train={len(y_train)}")

    best_score = -np.inf
    best_model = None
    best_name = None
    best_params = None
    cv_results_all = []

    logger.info("Starting model training with GridSearchCV across configured models...")
    for name, spec in (cfg.get("models") or {}).items():
        mtype = spec["type"]
        builder = MODEL_BUILDERS.get(mtype)
        if builder is None:
            logger.warning(f"Unknown model type '{mtype}', skipping.")
            continue
        model = builder()
        param_grid = spec.get("param_grid") or {}
        logger.info(f"GridSearchCV for {name} (type={mtype}) with grid: {param_grid}")
        gs = GridSearchCV(model, param_grid=param_grid, scoring="f1_macro", cv=5, n_jobs=-1, refit=True)
        logger.info(f"Fitting GridSearchCV for {name}...")
        gs.fit(X_train, y_train)
        logger.info(f"Completed GridSearchCV for {name}. Best f1_macro={gs.best_score_:.4f}; params={gs.best_params_}")
        cv_df = pd.DataFrame(gs.cv_results_)
        cv_df["model"] = name
        cv_results_all.append(cv_df)
        logger.info(f"Collected CV results for {name} with {len(cv_df)} rows.")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_
            best_params = gs.best_params_
            best_name = name

    # Save best model
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    meta_path = os.path.join(ARTIFACTS_DIR, "train_meta.json")
    save_joblib(model_path, best_model)
    meta = {
        "best_model": best_name,
        "best_score_cv_f1_macro": float(best_score),
        "best_params": best_params,
    }
    save_json(meta_path, meta)
    logger.info(f"Saved best model '{best_name}' to {model_path} and metadata to {meta_path}")

    # Save combined CV results
    if cv_results_all:
        all_df = pd.concat(cv_results_all, ignore_index=True)
        all_df.to_csv(os.path.join(ARTIFACTS_DIR, "cv_results.csv"), index=False)
        logger.info(f"Saved combined CV results to {os.path.join(ARTIFACTS_DIR, 'cv_results.csv')} with {len(all_df)} rows")

    logger.info(f"Training done. Best: {best_name} with f1_macro={best_score:.4f}")
    return model_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"))
    args = ap.parse_args()
    train(args.config)
