import numpy as np
import matplotlib.pyplot as plt

from mnist_loader import load_mnist


# =================================   DATASETS   ============================

np.random.seed(18)
data, target = load_mnist()
target = target.astype(int)   # đổi sang int trước khi so sánh

N = 2000
idx = np.random.choice(data.shape[0], N, replace=False)
X = data[idx] / 255.0      # shape (N, 784), mỗi mẫu là một hàng
Y = target[idx]            # shape (N,)

d = X.shape[1]                # 784


# ================================== THEM BIAS   ===========================
# Quy ước dùng xuyên suốt file: MỖI MẪU LÀ MỘT CỘT, Xbar có shape (d+1, N)

def add_bias(X_std):
    """X_std shape (d, M) -> thêm hàng toàn số 1 ở cuối -> shape (d+1, M)."""
    return np.concatenate((X_std, np.ones((1, X_std.shape[1]))), axis=0)


Xbar = add_bias(X.T)          # X.T: (784, N) -> Xbar: (785, N)
print("Xbar:", Xbar.shape, "| Y:", Y.shape)


# ================================== TACH TRAIN / TEST   ===========================
# Dữ liệu đã được lấy mẫu ngẫu nhiên nên cắt thẳng được.
# Mẫu nằm ở các CỘT của Xbar nên cắt theo trục cột.

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


# ================================   ONE VS ONE   ===========================


n_classes = 10
models = {} # w cua (i, j)
histories = {} # loss_history cua (i, j)


for i in range(n_classes):
    for j in range(i + 1, n_classes):
        mask = (Y_train == i) | (Y_train == j)      # chỉ lấy từ tập TRAIN

        Xbar_ij = Xbar_train[:, mask]
        Y_ij = (Y_train[mask] == j).astype(int) # vua phan loai mask, vua co : = 1 neu Y la j, =0 neu Y la i

        w_init = np.zeros(d + 1)
        w_ij, loss_history_ij = logistic_regression(w_init, Xbar_ij, Y_ij, eta, epochs)

        models[(i, j)] = w_ij
        histories[(i, j)] = loss_history_ij

        print(f"cặp {i}/{j}: {Xbar_ij.shape[1]} mẫu, loss cuối = {loss_history_ij[-1]:.4f}")


def predict_ovo(models, Xbar, n_classes):
    M = Xbar.shape[1] # so luonganh can du doan
    votes = np.zeros((M, n_classes)) # moi hang la 1 anh can du doan, 1 anh can du doanthuoc 10 lop
    prob_sum = np.zeros((M, n_classes)) # tong xac suat

    for (i, j), w in models.items():
        p = predict(w, Xbar)
        vote_j = p >= 0.5

        votes[vote_j, j] += 1
        votes[~vote_j, i] += 1
        prob_sum[:, j] += p
        prob_sum[:, i] += 1 - p

        # giai thich de hieu : dau tien 

        # votes = 
        # [[0, 0, 0],     ← mẫu 0
        # [0, 0, 0],      ← mẫu 1
        # [0, 0, 0],      ← mẫu 2
        # [0, 0, 0]]      ← mẫu 3

        # p = [0.3, 0.8, 0.7, 0.1] => 

        # vote_j  = [False, True, True, False]
        # ~vote_j = [True, False, False, True]

        # => di kiem votes[(true)][j] chon roi votes[(false)][i] chon
        # ... roi chon tk co xac suat cao nhat

    # prob_sum tối đa 9 (mỗi lớp chỉ có 9 cặp), chia 10 để luôn < 1 phiếu
    score = votes + prob_sum / n_classes
    return np.argmax(score, axis=1)


# ================================== DANH GIA   ===========================

y_pred_train = predict_ovo(models, Xbar_train, n_classes)
y_pred_test = predict_ovo(models, Xbar_test, n_classes)

print("Accuracy train:", np.mean(y_pred_train == Y_train))
print("Accuracy test :", np.mean(y_pred_test == Y_test))

# Độ chính xác theo từng chữ số trên tập test
for c in range(n_classes):
    m = Y_test == c
    if m.any():
        print(f"  số {c}: {np.mean(y_pred_test[m] == c):.3f} ({m.sum()} mẫu)")

# Đường loss của cả 45 bộ phân loại
for hist in histories.values():
    plt.plot(hist, alpha=0.4)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss theo epoch của 45 bộ phân loại One-vs-One")
plt.show()