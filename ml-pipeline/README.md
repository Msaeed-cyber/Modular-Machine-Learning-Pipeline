# Modular ML Pipeline (Preprocessing → Training → Evaluation)

A clean, reusable template for supervised **classification** using scikit-learn. It separates the workflow into modules and saves all artifacts for reuse.

## 📁 Project Structure
```text
ml-pipeline/
├─ data/
│  └─ dataset.csv                 # put your CSV/Excel here
├─ artifacts/                     # saved preprocessor, splits, models, metrics
├─ plots/                         # confusion matrix, ROC/PR curves, feature importance
├─ logs/
├─ src/
│  ├─ preprocessing.py            # loading, cleaning, encoding, scaling, splitting
│  ├─ train_model.py              # model training + GridSearchCV
│  ├─ evaluate.py                 # metrics + plots
│  └─ utils.py                    # logging + IO helpers
├─ config.yaml                    # configure dataset & hyperparameter grids
├─ main.py                        # orchestrates the full pipeline
├─ requirements.txt
└─ README.md
```

## 🔧 Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 📦 Add Your Dataset
- Place your file at `data/dataset.csv` (or use Excel).
- Update `config.yaml`:
  - `target_col`: the name of your label column.
  - `dataset_path`: path to your file.
  - `excel_sheet`: sheet name for Excel (or set to null for CSV).
  - Optionally list `id_columns` to drop.

#### Example datasets (open-source)
- UCI Machine Learning Repository – Heart Disease, Bank Marketing, etc.
  - https://archive.ics.uci.edu/
- OpenML – many tabular datasets (CSV-friendly)
  - https://www.openml.org/

## ▶️ Run the Pipeline
```bash
# Run everything (preprocess → train → evaluate)
python main.py --config config.yaml --step all

# Or run step-by-step
python main.py --step preprocess
python main.py --step train
python main.py --step evaluate
```

## 🧪 Models & Tuning
Configured in `config.yaml`:
- Logistic Regression, Decision Tree, Random Forest
- GridSearchCV over the provided parameter grids (scoring=`f1_macro` by default).

## 📊 Outputs
- `artifacts/preprocessor.joblib` – fitted ColumnTransformer
- `artifacts/train_data.joblib` and `artifacts/test_data.joblib`
- `artifacts/best_model.joblib` – best estimator
- `artifacts/train_meta.json` – best model summary
- `artifacts/cv_results.csv` – all CV rows
- `artifacts/evaluation_metrics.json` – Accuracy, Precision, Recall, F1 (macro)
- `artifacts/classification_report.json` – per-class metrics
- `plots/*.png` – confusion matrix, ROC/PR curves, feature importance

## ♻️ Reusability
- You can swap datasets by updating `config.yaml`.
- You can add more models by extending `models:` with a `type` and `param_grid`.

## 🛡 Best Practices
- Logging to `logs/pipeline.log`
- Reproducibility via `random_state` in `config.yaml`
- Meaningful exceptions on missing files/columns

## 📚 References
- scikit-learn: Pipeline, ColumnTransformer, model selection, and metrics
  - Pipeline & ColumnTransformer: https://scikit-learn.org/stable/modules/compose.html
  - Imputation: https://scikit-learn.org/stable/modules/impute.html
  - OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
  - GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
  - Classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
  - Permutation importance: https://scikit-learn.org/stable/modules/permutation_importance.html
