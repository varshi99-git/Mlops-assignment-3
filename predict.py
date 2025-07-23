from joblib import load
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = fetch_california_housing()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = load("model.joblib")


y_pred = model.predict(X_test)
print("First 10 predictions:", y_pred[:10])

r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2:.4f}")
