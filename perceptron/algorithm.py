from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt   # [SỬA 1] trước đây: import matplotlib as plt

np.random.seed(2)


# =============   DATASETS   ===============
# 15 hoc sinh, moi tk co so gio hoc va diem thi, du doan tk do co qua mon hay ko
# +1 la qua mon, -1 la rot me r


X = np.array([
    # --- 8 học sinh qua môn ---
    [6.0, 7.5], [7.5, 6.0], [5.0, 8.0], [8.0, 8.5],
    [4.5, 7.0], [6.5, 5.5], [9.0, 7.0], [7.0, 8.0],
    # --- 7 học sinh không qua môn ---
    [1.0, 3.0], [2.0, 5.0], [3.0, 2.5], [1.5, 6.0],
    [4.0, 3.5], [5.0, 4.0], [3.5, 5.5]
]).T 

y = np.array([1, 1, 1, 1, 1, 1, 1, 1,
              -1, -1, -1, -1, -1, -1, -1])

d, N = X.shape      # d = 2, N = 15


# ================   chuan hoa du lieu   =================
# z = (x - mean) / std


mu = X.mean(axis=1)       # shape (d,)  : trung bình của từng chiều
sigma = X.std(axis=1)     # shape (d,)  : độ lệch chuẩn của từng chiều


def standardize(X_raw):
    """X_raw có shape (d, M). Trừ/chia theo từng hàng."""
    return (X_raw - mu[:, np.newaxis]) / sigma[:, np.newaxis]


def add_bias(X_std):
    """Thêm HÀNG 1 vào cuối: mỗi cột x_bar = [z1, z2, 1]^T. Shape (d+1, M)."""
    return np.concatenate((X_std, np.ones((1, X_std.shape[1]))), axis=0)


Z = standardize(X)      # dữ liệu đã chuẩn hóa, shape (2, 15)
Xbar = add_bias(Z)      # thêm bias ở cuối,      shape (3, 15)


# ===================   perceptron   =======================



eta = 0.001

def loss_neural(w, Xbar, y, i):
    return -y[i] * np.dot(w, Xbar[:, i]) # thich thi tinh cho vui tai lat hoi lam xong ham mat mat = 0 me r 


def predict(w, Xbar):
    return np.where(w.dot(Xbar) >= 0, 1, -1)


def perceptron(w_init, Xbar, y, eta=0.001, max_epoch=1000):
    w = w_init.copy()

    for epoch in range(max_epoch):
        total_loss = 0
        n_errors = 0

        for i in np.random.permutation(N):
            x_i = Xbar[:, i]

            # tong tuyen tinh
            z = np.dot(w, x_i)      

            # ham kich hoat (sgn), toi uu w neu diem dang xet bi sai thoi (qua hoi quy logistic voi hoi quy softmax 
            # moi la cap nhat lien tuc du diem dang xet la dung)
            o = y[i] * z

            if o <= 0:# neu phan loai loi
                total_loss += loss_neural(w, Xbar, y, i)
                dw = y[i] * x_i
                w = w + eta * dw
                n_errors += 1

        if epoch % 100 == 0:
            print("Epoch %3d | số điểm sai = %2d | loss = %.5f"
                  % (epoch + 1, n_errors, total_loss))

        if n_errors == 0:  # không còn điểm sai -> hội tụ
            return w, epoch + 1

    return w, max_epoch


w_init = np.zeros(d + 1)  # khởi tạo w = 0  ->  [w1, w2, w0]
print("\nw khởi tạo =", w_init)


w, n_epochs = perceptron(w_init, Xbar, y, eta=eta)
w1, w2, w0 = w  # w0 là bias (phần tử cuối)


print("\nSố epoch để hội tụ:", n_epochs)
print("w = [w1, w2, w0] (trên dữ liệu đã chuẩn hóa) =", w)
print("Độ chính xác trên tập huấn luyện:",
      np.mean(predict(w, Xbar) == y) * 100, "%")


# Đổi w về đơn vị gốc (giờ học, điểm) để dễ diễn giải:
#   w1*(x1 - mu1)/s1 + w2*(x2 - mu2)/s2 + w0 = 0
b1 = w1 / sigma[0]
b2 = w2 / sigma[1]
b0 = w0 - b1 * mu[0] - b2 * mu[1]
print("Đường phân chia (đơn vị gốc): %.5f*giờ_học + %.5f*điểm_thi + %.5f = 0"
      % (b1, b2, b0))


# ==========================   hien thi w   ==============================


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Đơn vị gốc ---
ax = axes[0]
ax.plot(X[0, y == 1], X[1, y == 1], "b^", markersize=9, label="Qua môn (y = 1)")
ax.plot(X[0, y == -1], X[1, y == -1], "ro", markersize=8, label="Không qua (y = -1)")
x1 = np.linspace(0, 10, 100)
if abs(b2) > 1e-12:
    ax.plot(x1, -(b0 + b1 * x1) / b2, "g-", linewidth=2, label="Đường phân chia")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("Số giờ học")
ax.set_ylabel("Điểm thi")
ax.set_title("Dữ liệu gốc")
ax.legend()
ax.grid(alpha=0.3)

# --- Đã chuẩn hóa ---
ax = axes[1]
ax.plot(Z[0, y == 1], Z[1, y == 1], "b^", markersize=9, label="Qua môn (y = 1)")
ax.plot(Z[0, y == -1], Z[1, y == -1], "ro", markersize=8, label="Không qua (y = -1)")
z1 = np.linspace(-2.5, 2.5, 100)
if abs(w2) > 1e-12:
    ax.plot(z1, -(w0 + w1 * z1) / w2, "g-", linewidth=2, label="Đường phân chia")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlabel("Số giờ học (chuẩn hóa)")
ax.set_ylabel("Điểm thi (chuẩn hóa)")
ax.set_title("Dữ liệu đã chuẩn hóa (z-score)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("perceptron_qua_mon.png", dpi=120, bbox_inches="tight")
plt.show()