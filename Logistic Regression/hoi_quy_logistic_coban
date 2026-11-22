import numpy as np
import matplotlib.pyplot as plt

from mnist_loader import load_mnist


# =================================   DATASETS   ============================

np.random.seed(18)
data, target = load_mnist()
target = target.astype(int)   # đổi sang int trước khi so sánh

# Chỉ giữ lại số 0 và 1 (lọc TRƯỚC, rồi mới lấy mẫu ngẫu nhiên)
mask = np.isin(target, [0, 1])
data_01 = data[mask]
target_01 = target[mask]

N = 2000
idx = np.random.choice(data_01.shape[0], N, replace=False)
X = data_01[idx] / 255.0      # shape (N, 784), mỗi mẫu là một hàng
Y = target_01[idx]            # shape (N,)

d = X.shape[1]                # 784


# ================================== THEM BIAS   ===========================
# Quy ước dùng xuyên suốt file: MỖI MẪU LÀ MỘT CỘT, Xbar có shape (d+1, N)

def add_bias(X_std):
    """X_std shape (d, M) -> thêm hàng toàn số 1 ở cuối -> shape (d+1, M)."""
    return np.concatenate((X_std, np.ones((1, X_std.shape[1]))), axis=0)


Xbar = add_bias(X.T)          # X.T: (784, N) -> Xbar: (785, N)
print("Xbar:", Xbar.shape, "| Y:", Y.shape)


# ================================== HOI QUY LOGISTIC   ===========================

w_init = np.zeros(d + 1)      # shape (785,)
eta = 0.001
epochs = 2000


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def predict(w, Xbar):
    """Xác suất thuộc lớp 1 của mọi mẫu, shape (N,)."""
    return sigmoid(w @ Xbar)


def loss_neural(w, Xbar, Y):
    """Binary cross-entropy trung bình trên toàn bộ dữ liệu."""
    a = predict(w, Xbar)
    a = np.clip(a, 1e-12, 1 - 1e-12)       # tránh log(0)
    return -np.mean(Y * np.log(a) + (1 - Y) * np.log(1 - a))


def logistic_regression(w_init, Xbar, Y, epochs, eta):
    w = w_init.copy()
    N = Xbar.shape[1]                      # số mẫu nằm ở trục cột
    loss_history = []

    for epoch in range(epochs):
        for i in np.random.permutation(N):
            x_i = Xbar[:, i]               # một mẫu = một cột, shape (d+1,)
            y_i = Y[i]

            # tổng tuyến tính
            z = w @ x_i

            # hàm kích hoạt
            a = sigmoid(z)

            # gradient của mẫu i rồi cập nhật w
            dw = (a - y_i) * x_i
            w = w - eta * dw

        loss_history.append(loss_neural(w, Xbar, Y))

        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss = {loss_history[-1]:.4f}")

    return w, loss_history


w, loss_history = logistic_regression(w_init, Xbar, Y, epochs, eta)


# ================================== DANH GIA   ===========================

y_pred = (predict(w, Xbar) >= 0.5).astype(int)
print("Accuracy:", np.mean(y_pred == Y))

plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss theo epoch")
plt.show()