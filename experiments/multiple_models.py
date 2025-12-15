import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y)

mlflow.set_experiment("Model_Comparison")

for model, name in [
    (LinearRegression(), "LinearRegression"),
    (DecisionTreeRegressor(), "DecisionTree")
]:
    with mlflow.start_run(run_name=name):
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        rmse = mean_squared_error(yte, preds) ** 0.5
        mlflow.log_param("model", name)
        mlflow.log_metric("rmse", rmse)
