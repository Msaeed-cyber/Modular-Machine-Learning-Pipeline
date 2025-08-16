import os
import argparse
from src.preprocessing import preprocess
from src.train_model import train
from src.evaluate import evaluate

def main():
    ap = argparse.ArgumentParser(description="Modular ML Pipeline")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    ap.add_argument("--step", default="all", choices=["all", "preprocess", "train", "evaluate"], help="Which step to run")
    args = ap.parse_args()

    cfg_path = args.config

    if args.step in ["all", "preprocess"]:
        preprocess(cfg_path)
    if args.step in ["all", "train"]:
        train(cfg_path)
    if args.step in ["all", "evaluate"]:
        evaluate()

if __name__ == "__main__":
    main()
