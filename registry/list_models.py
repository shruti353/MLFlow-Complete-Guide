from mlflow.tracking import MlflowClient

client = MlflowClient()
model = "CaliforniaHousingModel"

for v in client.search_model_versions(f"name='{model}'"):
    print(v.version, v.aliases)
