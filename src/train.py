import logging
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score
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

def train() -> None:
    logger.info("Starting training pipeline...")
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    logger.info("Building unified model pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', get_preprocessor()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    logger.info("Training unified pipeline...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating unified pipeline on test set...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(exist_ok=True)
    
    logger.info("Saving unified pipeline artifact...")
    pipeline_path = models_dir / 'pipeline.pkl'
    joblib.dump(pipeline, pipeline_path)
    
    logger.info(f"Pipeline successfully saved to {pipeline_path}")

if __name__ == "__main__":
    train()
