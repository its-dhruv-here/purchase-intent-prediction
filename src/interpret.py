import logging
from pathlib import Path
import numpy as np

import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
import shap
import matplotlib.pyplot as plt

from src.data_processor import load_data, get_train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def interpret() -> None:
    logger.info("Starting interpretation pipeline...")
    
    X, y = load_data()
    _, X_test, _, _ = get_train_test_split(X, y)

    models_dir = PROJECT_ROOT / 'models'
    
    logger.info("Loading unified pipeline artifact...")
    pipeline_path = models_dir / 'pipeline.pkl'
    pipeline = joblib.load(pipeline_path)

    logger.info("Calculating SHAP values...")
    
    # SHAP's generic Explainer crashes on raw string columns (TypeError in np.isclose)
    # To bypass this library limitation while keeping our unified pipeline intact, 
    # we dynamically extract the preprocessor and classifier specifically for SHAP.
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform test set to purely numeric format
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    # Taking a random sample to keep TreeExplainer extremely fast
    X_test_sample = shap.utils.sample(X_test_transformed_df, 200)

    # Use native TreeExplainer which is much faster and accurate for XGBoost
    explainer = shap.TreeExplainer(classifier)
    
    logger.info("Evaluating SHAP on transformed test sample...")
    shap_values = explainer.shap_values(X_test_sample)

    logger.info("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    
    plot_path = PROJECT_ROOT / 'shap_summary.png'
    plt.savefig(plot_path)
    logger.info(f"SHAP summary plot saved as '{plot_path}'")

if __name__ == "__main__":
    interpret()
