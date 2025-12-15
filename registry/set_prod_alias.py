from mlflow.tracking import MlflowClient

client = MlflowClient()
model = "CaliforniaHousingModel"
version = "2"

for v in client.search_model_versions(f"name='{model}'"):
    if "prod" in v.aliases:
        client.delete_registered_model_alias(model, "prod")

client.set_registered_model_alias(model, "prod", version)
print("Production alias set")
