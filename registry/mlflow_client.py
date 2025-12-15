from mlflow.tracking import MlflowClient

client = MlflowClient()

print("Experiments:")
for exp in client.search_experiments():
    print(exp.name)

print("\nRegistered Models:")
for m in client.search_registered_models():
    print(m.name)
