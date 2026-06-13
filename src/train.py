import os
import json
import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# pyrefly: ignore [missing-import]
from imblearn.pipeline import Pipeline
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from src.data_processor import load_data, get_preprocessor, get_train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def save_confusion_matrix(y_true, y_pred, filepath, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def train() -> None:
    logger.info("Starting benchmarking and training pipeline...")
    
    reports_dir = PROJECT_ROOT / 'reports'
    figures_dir = reports_dir / 'figures'
    models_dir = PROJECT_ROOT / 'models'
    
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    preprocessor = get_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    best_f1 = -1
    best_model_name = ""
    best_pipeline = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        start_time = time.time()
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
        inference_time = time.time() - start_time

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"{name} - Accuracy: {acc:.4f}, Weighted F1: {f1_weighted:.4f}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "ROC AUC": roc_auc,
            "Train Time (s)": train_time,
            "Inference Time (s)": inference_time
        })

        # Save confusion matrix
        save_confusion_matrix(y_test, y_pred, figures_dir / f"cm_{name.replace(' ', '_').lower()}.png", f"Confusion Matrix - {name}")

        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_model_name = name
            best_pipeline = pipeline

    # Save benchmark results
    df_results = pd.DataFrame(results)
    df_results.to_csv(reports_dir / 'benchmark_results.csv', index=False)
    logger.info("Saved benchmark_results.csv")

    # Plot benchmark comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='Weighted F1', hue='Model', data=df_results, palette='viridis', legend=False)
    plt.title('Model Comparison - Weighted F1 Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figures_dir / 'benchmark_comparison.png', dpi=300)
    plt.close()

    # Save metrics JSON
    metrics = {
        "dataset": {
            "Total Rows": len(X),
            "Classes": len(np.unique(y)),
            "Train Samples": len(X_train),
            "Test Samples": len(X_test)
        },
        "benchmarks": results,
        "best_model": df_results[df_results['Model'] == best_model_name].to_dict(orient='records')[0]
    }
    with open(reports_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved metrics.json")

    logger.info(f"Saving best pipeline artifact ({best_model_name})...")
    pipeline_path = models_dir / 'pipeline.pkl'
    joblib.dump(best_pipeline, pipeline_path)
    logger.info(f"Best pipeline successfully saved to {pipeline_path}")

if __name__ == "__main__":
    train()
