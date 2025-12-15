from mlflow.tracking import MlflowClient

client = MlflowClient()
client.set_registered_model_alias("CaliforniaHousingModel", "prod", "1")
print("Rolled back to version 1")
