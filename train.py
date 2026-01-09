import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality.csv"
MODEL_PATH = "output/model/trained_model.pkl"
METRICS_PATH = "output/results/metrics.json"

os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

data = pd.read_csv(DATA_PATH)

X = data.drop("quality", axis=1)
y = data["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

joblib.dump(model, MODEL_PATH)

metrics = {"mse": mse, "r2_score": r2}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
