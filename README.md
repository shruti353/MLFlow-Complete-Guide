# MLFlow-Complete-Guide

End-to-end MLflow project demonstrating experiment tracking, model registry, versioning, alias-based production deployment, and inference pipelines using scikit-learn

This repository demonstrates end-to-end usage of MLflow:

- Experiment tracking
- Model comparison
- Model registry & versioning
- Alias-based production deployment (MLflow 3.x)
- Prediction on new data
- Registry automation using MlflowClient

## How to Run
1. Create venv and install requirements
2. Run training/training_models.py
3. Run mlflow ui
4. Set prod alias using registry/set_prod_alias.py
5. Run inference/predict.py
