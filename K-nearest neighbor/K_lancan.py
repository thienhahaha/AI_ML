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



# ====================================== K LAN CAN   =================================
# K láng giềng gần nhất KHÔNG có bước huấn luyện: "mô hình" chính là tập train.
# Để dự đoán một điểm mới: tìm k mẫu train gần nó nhất rồi bỏ phiếu theo nhãn của k mẫu đó.
# Hàng bias (toàn số 1) không ảnh hưởng khoảng cách vì hai điểm bất kỳ đều có 1 - 1 = 0 ở hàng đó.

n_classes = 10


def dist_pp(z, x):
    """Bình phương khoảng cách Euclid giữa 2 điểm (chưa lấy căn: thứ tự gần/xa không đổi)."""
    d = z - x.reshape(z.shape) # ep phai cung chieu neu co loi xay ra
    return np.sum(d * d)


def diss_ps_naive(z, Xbar):
    """Khoảng cách từ 1 điểm z (shape (d+1,)) tới MỌI mẫu của Xbar. Kết quả shape (1, N)."""
    N = Xbar.shape[1]
    res = np.zeros((1, N))                 # np.zeros nhận MỘT tuple shape, không phải (1, N) rời
    for i in range(N):
        res[0][i] = dist_pp(z, Xbar[:, i]) # mỗi mẫu là một CỘT nên Xbar[:, i], không phải Xbar[i]

    return res


def diss_ps_fast(z, Xbar):
    """Cùng kết quả với diss_ps_naive nhưng không vòng lặp.
    Công thức: ||z - x||² = ||z||² + ||x||² - 2·zᵀx

    Đầu vào:
        z    : 1 điểm, shape (d+1,)
        Xbar : N mẫu, mỗi mẫu một cột, shape (d+1, N)
    Trả về:
        res  : mảng shape (1, N), res[0][j] = bình phương khoảng cách từ z tới mẫu thứ j của Xbar
    """
    N = Xbar.shape[1]

    # ||z||²: một số vô hướng
    z_sq = np.sum(z * z)

    # ||x_j||² của từng mẫu j: shape (N,)
    X_sq = np.sum(Xbar * Xbar, axis=0)

    # zᵀx_j của từng mẫu j: shape (N,)
    zx = z @ Xbar

    # ghép lại theo công thức, shape (N,)
    dist = z_sq + X_sq - 2 * zx

    # đổi (N,) thành (1, N) cho giống diss_ps_naive
    res = dist.reshape(1, N)

    return res


def diss_ss_fast(Z, Xbar):
    """Khoảng cách từ MỌI điểm của tập Z tới MỌI mẫu của Xbar, tính một lần, không vòng lặp.

    Đầu vào:
        Z    : M điểm cần so sánh (ví dụ tập test), mỗi điểm một cột, shape (d+1, M)
        Xbar : N mẫu (ví dụ tập train), mỗi mẫu một cột, shape (d+1, N)
    Trả về:
        D    : MA TRẬN shape (M, N)
               D[m, j] = bình phương khoảng cách từ điểm thứ m của Z tới mẫu thứ j của Xbar
               (hàng m của D chứa khoảng cách từ điểm m tới cả N mẫu)
    """
    M = Z.shape[1]
    N = Xbar.shape[1]

    # ||z_m||² của từng điểm m: shape (M,), rồi xếp thành CỘT shape (M, 1)
    Z_sq = np.sum(Z * Z, axis=0)
    Z_sq = Z_sq.reshape(M, 1)

    # ||x_j||² của từng mẫu j: shape (N,), rồi xếp thành HÀNG shape (1, N)
    X_sq = np.sum(Xbar * Xbar, axis=0)
    X_sq = X_sq.reshape(1, N)

    # z_mᵀx_j của MỌI cặp (m, j): shape (M, N)
    ZX = Z.T @ Xbar

    # D[m, j] = ||z_m||² + ||x_j||² - 2·z_mᵀx_j
    # (M, 1) + (1, N) - (M, N): NumPy tự "kéo giãn" cột và hàng thành (M, N) (broadcasting)
    D = Z_sq + X_sq - 2 * ZX

    return D


# Kiểm tra 2 cách tính cho cùng kết quả
z0 = Xbar_test[:, 0]
print("naive == fast:", np.allclose(diss_ps_naive(z0, Xbar_train), diss_ps_fast(z0, Xbar_train)))


def knn_predict(Xbar_train, Y_train, Xbar_test, k):
    """Dự đoán nhãn cho từng mẫu test bằng cách bỏ phiếu của k láng giềng gần nhất.

    Trả về:
        y_pred : mảng nguyên shape (M,), y_pred[m] là nhãn dự đoán (0..9) của mẫu test thứ m
    """
    M = Xbar_test.shape[1]

    # D[m, j] = bình phương khoảng cách từ mẫu test m tới mẫu train j, shape (M, N_train)
    D = diss_ss_fast(Xbar_test, Xbar_train)

    # sắp xếp từng hàng của D tăng dần, lấy chỉ số của k mẫu train gần nhất, shape (M, k)
    order = np.argsort(D, axis=1) # khac voi sort, argsort la sap xep tra ve index chu ko phai value
    # vidu : sort (1, 3, 5 ,4, 6) thanh (1, 3, 4, 5, 6)
    # argsort : thanh (0, 1, 3, 2, 4)
    nearest = order[:, :k]

    # nhãn của k láng giềng, shape (M, k)
    labels = Y_train[nearest]

    # bỏ phiếu cho từng mẫu test
    y_pred = np.zeros(M, dtype=int)
    for m in range(M):
        votes = np.bincount(labels[m], minlength=n_classes)   # votes[c] = số láng giềng có nhãn c, shape (10,)
        y_pred[m] = np.argmax(votes)                          # nhãn nhiều phiếu nhất (hòa thì chọn nhãn nhỏ hơn)

    return y_pred


# ================================== DANH GIA   ===========================

ks = [1, 3, 5, 7, 9, 11] # chon so luong k (chon cai nao xin nhat)
accs = [] # accuracy ung voi tung k
for k in ks:
    y_pred_test = knn_predict(Xbar_train, Y_train, Xbar_test, k)
    acc = np.mean(y_pred_test == Y_test)
    accs.append(acc)
    print(f"k = {k:2d}: accuracy test = {acc:.4f}")

# Độ chính xác theo từng chữ số với k tốt nhất
k_best = ks[int(np.argmax(accs))]
y_pred_test = knn_predict(Xbar_train, Y_train, Xbar_test, k_best)
print(f"k tốt nhất = {k_best}")
for c in range(n_classes):
    m = Y_test == c
    if m.any():
        print(f"  số {c}: {np.mean(y_pred_test[m] == c):.3f} ({m.sum()} mẫu)")

plt.plot(ks, accs, marker="o")
plt.xlabel("k")
plt.ylabel("accuracy test")
plt.title("K láng giềng gần nhất: accuracy theo k")
plt.show()