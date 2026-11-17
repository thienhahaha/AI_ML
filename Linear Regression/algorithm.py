import numpy as np
import matplotlib.pyplot as plt

print("NumPy:", np.__version__)
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

# Thêm bias: mỗi cột của X_bias là một điểm dữ liệu [x_scaled, 1]^T
X_bias = np.vstack((X_scaled.T, np.ones(X.shape[0])))   # shape (2, N)


# ==============================
# 3. CÁC HÀM CƠ BẢN (viết rõ ràng)
# ==============================

def predict_one(w, x_i):
    """Dự đoán cho MỘT điểm dữ liệu: y_pred_i = w^T x_i"""
    return (w.T @ x_i).item()


def loss_one(w, x_i, y_i):
    """
    Mất mát của MỘT điểm dữ liệu:

        l_i = 1/2 * (w^T x_i - y_i)^2
    """
    y_pred_i = predict_one(w, x_i)
    error_i = y_pred_i - y_i
    return 0.5 * error_i ** 2


def compute_loss(w, X_bias, y):
    """
    Hàm mất mát trên TOÀN BỘ tập dữ liệu (trung bình của mất mát từng điểm):

        L(w) = 1/N * sum_{i=1..N} 1/2 * (w^T x_i - y_i)^2
    """
    N = X_bias.shape[1]
    total = 0.0

    for i in range(N):                 # duyệt từng điểm dữ liệu
        x_i = X_bias[:, i:i+1]         # shape (2, 1)
        y_i = y[i]
        total += loss_one(w, x_i, y_i) # cộng dồn mất mát của điểm i

    return total / N                   # chia N để lấy trung bình


def gradient_one(w, x_i, y_i):
    """
    Gradient của mất mát MỘT điểm theo w:

        d l_i / d w = (w^T x_i - y_i) * x_i
    """
    error_i = predict_one(w, x_i) - y_i
    return error_i * x_i


# ==============================
# 4. INITIALIZE PARAMETERS
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

N = X_bias.shape[1]


# ==============================
# 5. STOCHASTIC GRADIENT DESCENT
# ==============================

for epoch in range(epochs):

    # --- Cập nhật w sau MỖI điểm dữ liệu ---
    for i in range(N):
        x_i = X_bias[:, i:i+1]
        y_i = y[i]

        dw = gradient_one(w, x_i, y_i)
        w = w - eta * dw

    # --- Tính loss trên toàn bộ dữ liệu sau mỗi epoch ---
    loss = compute_loss(w, X_bias, y)

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

    # --- Early stopping ---
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
# 6. FINAL RESULT
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
# 7. VISUALIZATION
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