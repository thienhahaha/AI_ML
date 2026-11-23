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



# ================================== ONE-HOT   ===========================
# Nhãn 3 -> cột [0,0,0,1,0,0,0,0,0,0]. Y_onehot có shape (C, M): mỗi MẪU là một CỘT.

n_classes = 10


def one_hot(Y, n_classes):
    Y_onehot = np.zeros((n_classes, Y.shape[0]))
    Y_onehot[Y, np.arange(Y.shape[0])] = 1        # hàng = nhãn, cột = mẫu
    return Y_onehot


Y_train_oh = one_hot(Y_train, n_classes)          # (10, 1600)
Y_test_oh = one_hot(Y_test, n_classes)            # (10, 400)
print("one-hot:", Y_train_oh.shape, Y_test_oh.shape)



# ================================== HOI QUY SOFTMAX   ===========================


w_init = np.zeros((d + 1, n_classes))      # shape (785, 10)
eta = 0.001
epochs = 2000


def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)       # trừ max để exp không bị tràn số
    e = np.exp(Z)
    return e / np.sum(e, axis=0, keepdims=True)


def predict(W, Xbar):
    return softmax(W.T @ Xbar)


def loss_neural(W, Xbar, Y):
    A = predict(W, Xbar)
    A = np.clip(A, 1e-12, 1)                       # tránh log(0)
    return -np.mean(np.sum(Y * np.log(A), axis=0))


def softmax_regression(w_init, Xbar, Y, eta, epochs):
    w = w_init.copy()
    N = Xbar.shape[1]
    loss_history = []

    for epoch in range(epochs):
        for i in np.random.permutation(N):
            x_i = Xbar[:, i]
            y_i = Y[:, i]

            # tong tuyen tinh
            z = w.T @ x_i

            # ham kich hoat
            a = softmax(z)

            # toi uu w
            dw = np.outer(x_i, a - y_i)
            w = w - eta * dw

        loss_history.append(loss_neural(w, Xbar, Y))

        if epoch % 100 == 0:
            print(f"epoch {epoch}: loss = {loss_history[-1]:.4f}")

    return w, loss_history


w, loss_history = softmax_regression(w_init, Xbar_train, Y_train_oh, eta, epochs)


# ================================== DANH GIA   ===========================

y_pred_train = np.argmax(predict(w, Xbar_train), axis=0)    # lớp có xác suất lớn nhất của mỗi cột, shape (1600,)
y_pred_test = np.argmax(predict(w, Xbar_test), axis=0)      # shape (400,)

print("Accuracy train:", np.mean(y_pred_train == Y_train))
print("Accuracy test :", np.mean(y_pred_test == Y_test))

# Độ chính xác theo từng chữ số trên tập test
for c in range(n_classes):
    m = Y_test == c
    if m.any():
        print(f"  số {c}: {np.mean(y_pred_test[m] == c):.3f} ({m.sum()} mẫu)")

plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss theo epoch (softmax regression)")
plt.show()