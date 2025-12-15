import mlflow

mlflow.set_experiment("Basic_Tracking")

with mlflow.start_run():
    mlflow.log_param("param1", 10)
    mlflow.log_metric("metric1", 0.95)
