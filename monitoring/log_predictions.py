import mlflow

with mlflow.start_run(run_name="monitoring"):
    mlflow.log_metric("prediction", 4.22)
