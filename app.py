# pyrefly: ignore [missing-import]
import logging
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import gradio as gr
import pandas as pd
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

def load_pipeline():
    logger.info("Loading unified pipeline artifact...")
    models_dir = PROJECT_ROOT / 'models'
    pipeline_path = models_dir / 'pipeline.pkl'
    return joblib.load(pipeline_path)

def create_app() -> gr.Interface:
    pipeline = load_pipeline()

    def predict_conversion(*args: Any) -> str:
        logger.info("Received prediction request.")
        feature_names = [
            'Administrative', 'Administrative_Duration', 'Informational', 
            'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
            'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay',
            'Month', 'OperatingSystems', 'Browser', 'Region', 
            'TrafficType', 'VisitorType', 'Weekend'
        ]
        
        input_data = dict(zip(feature_names, args))
        df = pd.DataFrame([input_data])
        
        # Predict probability directly using the unified pipeline!
        prob = pipeline.predict_proba(df)[0, 1]
        
        logger.info(f"Prediction complete. Conversion Probability: {prob * 100:.2f}%")
        return f"Conversion Probability: {prob * 100:.2f}%"

    inputs = [
        gr.Number(label='Administrative', value=0),
        gr.Number(label='Administrative_Duration', value=0.0),
        gr.Number(label='Informational', value=0),
        gr.Number(label='Informational_Duration', value=0.0),
        gr.Number(label='ProductRelated', value=0),
        gr.Number(label='ProductRelated_Duration', value=0.0),
        gr.Number(label='BounceRates', value=0.0),
        gr.Number(label='ExitRates', value=0.0),
        gr.Number(label='PageValues', value=0.0),
        gr.Number(label='SpecialDay', value=0.0),
        gr.Dropdown(choices=['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], label='Month', value='May'),
        gr.Number(label='OperatingSystems', value=1),
        gr.Number(label='Browser', value=1),
        gr.Number(label='Region', value=1),
        gr.Number(label='TrafficType', value=1),
        gr.Dropdown(choices=['Returning_Visitor', 'New_Visitor', 'Other'], label='VisitorType', value='Returning_Visitor'),
        gr.Checkbox(label='Weekend', value=False)
    ]

    app = gr.Interface(
        fn=predict_conversion,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction Result"),
        title="E-Commerce Conversion Predictor",
        description="Predicts the likelihood of a customer completing a purchase (not abandoning the cart) based on their session data."
    )
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
