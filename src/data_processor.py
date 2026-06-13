import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine project root based on this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_data(filepath: Path | str = PROJECT_ROOT / "online_shoppers_intention.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Loads dataset and returns features X and target y."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df['Revenue'] = df['Revenue'].astype(int)
    y = df['Revenue']
    X = df.drop('Revenue', axis=1)
    logger.info(f"Data loaded successfully. Features shape: {X.shape}")
    return X, y

def get_preprocessor() -> ColumnTransformer:
    """Builds and returns the scikit-learn ColumnTransformer."""
    logger.info("Building preprocessing pipeline...")
    num_features = ['Administrative', 'Administrative_Duration', 'Informational', 
                    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    cat_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 
                    'TrafficType', 'VisitorType', 'Weekend']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    return preprocessor

def get_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into train and test sets."""
    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
