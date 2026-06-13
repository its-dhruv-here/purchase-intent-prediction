# Feature Engineering & Preprocessing Architecture

## Overview
This document details the data engineering pipelines established for the Session Purchase Intent Engine. Given the heterogeneous nature of e-commerce session data, rigorous preprocessing was required to normalize feature distributions, encode high-cardinality nominal features, and address significant class imbalances inherent to conversion datasets.

## Pipeline Components

### 1. Numerical Feature Scaling
**Algorithm**: `StandardScaler` (Z-score Normalization)
**Target Features**: 
- Session Durations (`Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration`)
- Traversal Counts (`Administrative`, `Informational`, `ProductRelated`)
- Behavioral Metrics (`BounceRates`, `ExitRates`, `PageValues`)
- Temporal Proximity (`SpecialDay`)

**Engineering Rationale**: 
Duration and count metrics in web analytics follow highly right-skewed, heavy-tailed distributions. Z-score normalization centers these variables to a zero mean and unit variance. This mitigates the risk of gradient dominance by features with inherently larger magnitudes (e.g., `ProductRelated_Duration` vs `BounceRates`), ensuring balanced feature weight optimization during model convergence, particularly for gradient-based and regularized linear classifiers.

### 2. Categorical Variable Encoding
**Algorithm**: `OneHotEncoder`
**Target Features**:
- `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`

**Engineering Rationale**:
Nominal variables such as `Browser` or `TrafficType` lack intrinsic ordinality. Feeding these directly as integers introduces a false distance metric to the estimator (e.g., implying Browser 3 is "greater" than Browser 2). One-hot encoding transforms these nominal classes into a high-dimensional sparse boolean matrix, strictly isolating the marginal probability shift associated with each distinct category without implying hierarchy. The transformer is configured to gracefully ignore unknown categories during inference, preventing pipeline crashes when encountering novel user-agents in production.

### 3. Class Imbalance Rectification
**Algorithm**: `SMOTE` (Synthetic Minority Over-sampling Technique)
**Implementation Layer**: Pipeline Resampling Node

**Engineering Rationale**:
E-commerce conversion rates typically hover between 1% and 5%, resulting in a severe structural target class imbalance. Models trained on the raw distribution naturally collapse to predicting the majority class (Abandonment) to minimize global loss, resulting in unacceptable recall for the minority class (Purchase).
By integrating SMOTE exclusively on the training manifold *after* the train-test split, we synthesize high-fidelity interpolations of the minority class feature space. This shifts the decision boundary, trading a minor reduction in precision for a substantial, required gain in recall—ensuring the engine reliably identifies actionable, high-intent sessions without polluting the validation set.

## Extensibility
The preprocessing topology is fully encapsulated within a scikit-learn `ColumnTransformer` embedded inside an `imblearn` unified pipeline. This architecture guarantees that data transformation parameters (means, variances, categorical mappings) learned during the fit phase are immutably serialized and identical during real-time inference, structurally eliminating training-serving skew.
