import numpy as np
import matplotlib.pyplot as plt

from mnist_loader import load_mnist


# =================================   DATASETS   ============================

np.random.seed(18)
data, target = load_mnist()
target = target.astype(int)   # đổi sang int trước khi so sánh

N = 2000
idx = np.random.choice(data.shape[0], N, replace=False)
X = (data[idx] / 255.0).T  # data[idx] là (N, 784) nên chuyển vị: X có shape (784, N), mỗi mẫu là một CỘT
Y = target[idx]            # shape (N,)

d = X.shape[0]                # 784 (số chiều = số hàng của X)


# ================================== THEM BIAS   ===========================
# Quy ước dùng xuyên suốt file: MỖI MẪU LÀ MỘT CỘT, Xbar có shape (d+1, N)

def add_bias(X_std):
    """X_std shape (d, M) -> thêm hàng toàn số 1 ở cuối -> shape (d+1, M)."""
    return np.concatenate((X_std, np.ones((1, X_std.shape[1]))), axis=0)


Xbar = add_bias(X)            # X: (784, N) -> Xbar: (785, N)
print("Xbar:", Xbar.shape, "| Y:", Y.shape)



# ================================== TACH TRAIN / TEST   ===========================


n_train = int(0.8 * N)                        # 1600

Xbar_train, Xbar_test = Xbar[:, :n_train], Xbar[:, n_train:]    # (785, 1600), (785, 400)
Y_train, Y_test = Y[:n_train], Y[n_train:]                      # (1600,), (400,)
print("train:", Xbar_train.shape, Y_train.shape, "| test:", Xbar_test.shape, Y_test.shape)



# ================================== HOI QUY TUYEN TINH   ===========================



w_init = np.zeros(d + 1)      # shape (785,)
eta = 0.001
epochs = 2000


def predict(w, Xbar):
    return w @ Xbar


def loss(w, Xbar, Y):
    return 0.5 * np.mean((predict(w, Xbar) - Y) ** 2)


def linear_regression(w_init, Xbar, Y, eta, epochs):
    w = w_init.copy()
    N = Xbar.shape[1]                     
    loss_history = []

    for epoch in range(epochs):
        for i in np.random.permutation(N):
            x_i = Xbar[:, i]         
            y_i = Y[i]

            # tong tuyen tinh
            z = predict(w, x_i)

            # toi uu w
            dw = (z - y_i) * x_i  
            w = w - eta * dw

        loss_history.append(loss(w, Xbar, Y))

        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss = {loss_history[-1]:.4f}")

    return w, loss_history


w, loss_history = linear_regression(w_init, Xbar_train, Y_train, eta, epochs)



# ================================== DANH GIA   ===========================



y_pred_train = predict(w, Xbar_train)
y_pred_test = predict(w, Xbar_test)

print("MSE train:", np.mean((y_pred_train - Y_train) ** 2))
print("MSE test :", np.mean((y_pred_test - Y_test) ** 2))

# Mốc so sánh: luôn đoán giá trị trung bình của Y_train
print("MSE khi luôn đoán trung bình:", np.mean((Y_test - Y_train.mean()) ** 2))

# Nhãn là chữ số 0..9 nên có thể làm tròn để so với nhãn thật (chỉ mang tính tham khảo)
acc_test = np.mean(np.clip(np.round(y_pred_test), 0, 9) == Y_test)
print("Accuracy test sau khi làm tròn:", acc_test)

plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss (MSE / 2)")
plt.title("Loss theo epoch")
plt.show()