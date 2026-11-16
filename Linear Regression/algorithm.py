
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import sklearn

print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Matplotlib:", plt.matplotlib.__version__)

# ==============================
# 1. DATASET
# ==============================

X = np.array([
    [147, 150, 153, 158, 163, 165, 168,
     170, 173, 175, 178, 180, 183]
]).T

y = np.array([
    49, 50, 51, 54, 58, 59, 60,
    62, 63, 64, 66, 67, 68
])

# ==============================
# 2. NORMALIZE DATA
# ==============================

X_mean = X.mean()
X_std = X.std()

X_scaled = (X - X_mean) / X_std

# Add bias
X_bias = np.vstack((X_scaled.T, np.ones(X.shape[0])))

# ==============================
# 3. INITIALIZE PARAMETERS
# ==============================

w = np.zeros((2, 1))

eta = 0.01
epochs = 10000

# Early stopping parameters
patience = 30
min_delta = 1e-10

best_loss = float("inf")
best_w = w.copy()
wait = 0

# ==============================
# 4. STOCHASTIC GRADIENT DESCENT
# ==============================

for epoch in range(epochs):

    # Train through all data points
    for i in range(X_bias.shape[1]):

        x_i = X_bias[:, i:i+1]
        y_i = y[i]

        # Forward
        y_pred = w.T @ x_i

        # Error
        error = y_pred.item() - y_i

        # Gradient
        dw = error * x_i

        # Update weights
        w = w - eta * dw

    # ==============================
    # 5. CALCULATE LOSS AFTER EPOCH
    # ==============================

    y_pred_all = (w.T @ X_bias).ravel()

    errors = y_pred_all - y

    loss = np.mean(errors ** 2)

    # Check numerical stability
    if not np.isfinite(loss) or not np.all(np.isfinite(w)):
        print("\nNumerical instability detected!")
        break

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Loss = {loss:.10f}, "
            f"w = {w.ravel()}"
        )

    # ==============================
    # 6. EARLY STOPPING
    # ==============================

    if loss < best_loss - min_delta:

        best_loss = loss
        best_w = w.copy()
        wait = 0

    else:
        wait += 1

    if wait >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# Restore best weights
w = best_w

# ==============================
# 7. FINAL RESULT
# ==============================

print("\nFinal w (normalized):")
print(w)

print("\nFinal Loss:")
print(best_loss)

# Convert weights back to original scale
slope = w[0, 0] / X_std
intercept = w[1, 0] - slope * X_mean

print("\nRegression equation:")
print(f"y = {slope:.6f}x + ({intercept:.6f})")

# ==============================
# 9. VISUALIZATION
# ==============================

plt.scatter(X, y, label="Actual data")

x_line = np.linspace(145, 185, 100).reshape(-1, 1)

y_line = slope * x_line + intercept

plt.plot(
    x_line,
    y_line,
    color="red",
    label="Regression line"
)

plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Linear Regression using SGD")

plt.legend()
plt.grid(True)
plt.show()
