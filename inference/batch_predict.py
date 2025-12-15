import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/CaliforniaHousingModel@prod")
df = pd.read_csv("data/sample_input.csv")
df["prediction"] = model.predict(df)
print(df)
