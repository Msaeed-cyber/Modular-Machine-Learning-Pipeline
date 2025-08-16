import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance
from .utils import get_logger, load_joblib, save_json

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def evaluate() -> str:
    logger = get_logger("evaluate")
    model = load_joblib(os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
    test_data = load_joblib(os.path.join(ARTIFACTS_DIR, "test_data.joblib"))
    preprocessor = load_joblib(os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    X_test, y_test = test_data["X"], test_data["y"]
    logger.info(f"Loaded artifacts for evaluation. X_test={getattr(X_test, 'shape', None)}, y_test={len(y_test)}")

    y_pred = model.predict(X_test)
    logger.info("Predictions computed for test set.")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")

    # ROC / PR if probabilities are available
    roc_path, pr_path = None, None
    if hasattr(model, "predict_proba"):
        try:
            # Handle binary case
            RocCurveDisplay.from_estimator(model, X_test, y_test)
            roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
            plt.title("ROC Curve")
            plt.savefig(roc_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved ROC curve to {roc_path}")

            PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
            pr_path = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
            plt.title("Precision-Recall Curve")
            plt.savefig(pr_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved PR curve to {pr_path}")
        except Exception as e:
            logger.warning(f"Could not plot ROC/PR curves: {e}")

    # Permutation importance (model-agnostic)
    try:
        logger.info("Computing permutation importance (n_repeats=5)...")
        result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        sorted_idx = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))

        # Get feature names from preprocessor, fallback to indices
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = np.array([f"f{i}" for i in range(X_test.shape[1])])

        top_names = feature_names[sorted_idx][:top_k]
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_k), importances[sorted_idx][:top_k][::-1])
        plt.yticks(range(top_k), top_names[::-1])
        plt.title("Permutation Feature Importance (Top 20)")
        plt.xlabel("Importance")
        fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
        plt.tight_layout()
        plt.savefig(fi_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved permutation feature importance to {fi_path}")
        # Log top-10 permutation importances to console for CLI visibility
        top_show = min(10, top_k)
        top_pairs = list(zip(top_names[:top_show], importances[sorted_idx][:top_show]))
        logger.info("Top permutation importances (name, score): " + ", ".join([f"{n}: {v:.4f}" for n, v in top_pairs]))
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        fi_path = None

    # Model-based feature importance/coefficient plot when available
    try:
        if hasattr(model, "feature_importances_"):
            importances_mb = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            importances_mb = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            importances_mb = None

        if importances_mb is not None:
            try:
                feature_names = preprocessor.get_feature_names_out()
            except Exception:
                feature_names = np.array([f"f{i}" for i in range(X_test.shape[1])])

            sorted_idx_mb = np.argsort(importances_mb)[::-1]
            top_k_mb = min(20, len(importances_mb))
            top_names_mb = feature_names[sorted_idx_mb][:top_k_mb]
            plt.figure(figsize=(10, 6))
            plt.barh(range(top_k_mb), importances_mb[sorted_idx_mb][:top_k_mb][::-1])
            plt.yticks(range(top_k_mb), top_names_mb[::-1])
            plt.title("Model-based Feature Importance / Coefficients (Top 20)")
            plt.xlabel("Importance (abs)")
            fi_model_path = os.path.join(PLOTS_DIR, "model_feature_importance.png")
            plt.tight_layout()
            plt.savefig(fi_model_path, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved model-based feature importance to {fi_model_path}")
            # Log top-10 model-based importance/coefs
            top_show_mb = min(10, top_k_mb)
            top_pairs_mb = list(zip(top_names_mb[:top_show_mb], importances_mb[sorted_idx_mb][:top_show_mb]))
            logger.info("Top model-based importances (name, score): " + ", ".join([f"{n}: {v:.4f}" for n, v in top_pairs_mb]))
    except Exception as e:
        logger.warning(f"Model-based feature importance failed: {e}")

    # Save metrics
    save_json(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), metrics)
    save_json(os.path.join(ARTIFACTS_DIR, "classification_report.json"), report)
    logger.info(f"Evaluation done: {metrics}")
    return cm_path

if __name__ == "__main__":
    evaluate()
