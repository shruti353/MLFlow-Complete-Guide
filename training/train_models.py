import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("California_Housing_Experiment")

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y)

with mlflow.start_run(run_name="Random_Forest"):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    rmse = mean_squared_error(yte, preds) ** 0.5

    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, name="model")
