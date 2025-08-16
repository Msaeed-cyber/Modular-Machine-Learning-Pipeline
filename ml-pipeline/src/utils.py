import os
import json
import logging
from datetime import datetime
from typing import Any, Dict
import joblib

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

        # console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # file
        fh = logging.FileHandler(os.path.join(LOG_DIR, "pipeline.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_joblib(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: str) -> Any:
    return joblib.load(path)
