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


def logistic_regression(w_init, Xbar, Y, eta, epochs):
    w = w_init.copy()
    N = Xbar.shape[1]                      # số mẫu nằm ở trục cột (không phải shape[0])
    loss_history = []

    for epoch in range(epochs):
        for i in np.random.permutation(N):
            x_i = Xbar[:, i]
            y_i = Y[i]

            # tong tuyen tinh
            z = w @ x_i

            # ham kich hoat
            a = sigmoid(z)

            # toi uu w
            dw = (a - y_i) * x_i
            w = w - eta * dw

        loss_history.append(loss_neural(w, Xbar, Y))

        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss = {loss_history[-1]:.4f}")

    return w, loss_history


# ================================   ONE VS REST   ===========================


n_classes = 10
models = {} # w cua (i vs rest)
histories = {} # loss_history cua (i vs rest)


for i in range(0, n_classes):
    Y_i = (Y_train == i).astype(int) # Y_i[m] = 1 neu Y_train[m] == i (mau thu m la so i), con lai = 0

    w_init = np.zeros(d + 1)
    w_i, loss_history_i = logistic_regression(w_init, Xbar_train, Y_i, eta, epochs)   # Xbar_train (1600 mau) khop voi Y_i (1600 nhan)

    models[i] = w_i
    histories[i] = loss_history_i

    n_pos = Y_i.sum()
    print(f"lớp {i} vs rest: {n_pos} mẫu dương, {len(Y_i) - n_pos} mẫu âm, loss cuối = {loss_history_i[-1]:.4f}")


def predict_ovr(models, Xbar, n_classes):
    M = Xbar.shape[1] # so luong anh can du doan
    scores = np.zeros((M, n_classes)) # moi hang la 1 anh can du doan

    for c in range(n_classes):
        scores[:, c] = models[c] @ Xbar    

    return np.argmax(scores, axis=1)


# ================================== DANH GIA   ===========================


y_pred_train = predict_ovr(models, Xbar_train, n_classes)
y_pred_test = predict_ovr(models, Xbar_test, n_classes)

print("Accuracy train:", np.mean(y_pred_train == Y_train))
print("Accuracy test :", np.mean(y_pred_test == Y_test))

# Độ chính xác theo từng chữ số trên tập test
for c in range(n_classes):
    m = Y_test == c
    if m.any():
        print(f"  số {c}: {np.mean(y_pred_test[m] == c):.3f} ({m.sum()} mẫu)")

# Đường loss của 10 bộ phân loại
for c, hist in histories.items():
    plt.plot(hist, label=f"{c} vs rest")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss theo epoch của 10 bộ phân loại One-vs-Rest")
plt.legend()
plt.show()