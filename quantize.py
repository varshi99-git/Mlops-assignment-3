import os
import torch
import joblib
import numpy as np
import torch.nn as nn
from joblib import dump
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# --- Paths ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)


# --- Load Dataset (Regression) ---
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- Train Sklearn Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
dump(model, MODEL_PATH)
print(f"✅ Trained and saved model to {MODEL_PATH}")

# Extract coefficients
coef = model.coef_            # Shape: [n_features]
intercept = model.intercept_  # Shape: scalar

# --- Save unquantized parameters ---
unquant_params = {"coef": coef, "intercept": intercept}
dump(unquant_params, "unquant_params.joblib")


# --- Quantization Functions ---
def quantize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    q = np.round((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    return q, arr_min, arr_max


def dequantize(q, arr_min, arr_max):
    return q.astype(np.float32) / 255 * (arr_max - arr_min) + arr_min


# --- Quantize weights ---
q_coef, coef_min, coef_max = quantize(coef)
q_intercept, intercept_min, intercept_max = quantize(np.array([intercept]))


# Save quantized parameters
quant_params = {
    "coef": q_coef,
    "intercept": q_intercept,
    "coef_min": coef_min,
    "coef_max": coef_max,
    "intercept_min": intercept_min,
    "intercept_max": intercept_max
}
dump(quant_params, "quant_params.joblib")


# --- PyTorch Linear Regression Model ---
class LinearRegTorch(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.linear(x)


# Dequantize parameters and load into PyTorch model
dequant_coef = dequantize(q_coef, coef_min, coef_max).astype(np.float32)
dequant_intercept = dequantize(q_intercept, intercept_min, intercept_max).astype(np.float32)[0]

net = LinearRegTorch(X.shape[1])
net.linear.weight.data = torch.tensor([dequant_coef])  # shape should be [1, n_features]
net.linear.bias.data = torch.tensor([dequant_intercept])  # shape [1]

# Save pytorch model
torch.save(net.state_dict(), "quantized_pytorch.pt")
print(" Saved quantized model as quantized_pytorch.pt")


# --- Inference and Evaluation ---
net.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds = net(X_tensor).detach().cpu().numpy().flatten()  # Shape [n_samples]

r2 = r2_score(y_test, preds)
print(f" Quantized PyTorch R2 score: {r2:.4f}")
