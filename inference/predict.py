import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model(
    "models:/CaliforniaHousingModel@prod"
)

data = pd.DataFrame([[8.3,41,6.98,1.02,322,2.55,37.88,-122.23]])
pred = model.predict(data)
print("Prediction:", pred)
