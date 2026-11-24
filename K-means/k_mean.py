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



# ==================================== K MEANS   =====================================
# K-means là thuật toán KHÔNG giám sát: Y_train không được dùng để gom cụm.
# Quy ước: mỗi mẫu là một CỘT, mỗi tâm cụm cũng là một CỘT: centroids có shape (d+1, K).


n_classes = 10
K = 10


def kmeans_init_random(Xbar, k):
    """Chọn ngẫu nhiên k mẫu (k CỘT) của Xbar làm tâm cụm ban đầu. Trả về shape (d+1, k)."""
    cols = np.random.choice(Xbar.shape[1], k, replace=False)   # k chỉ số cột khác nhau
    return Xbar[:, cols]


def kmeans_assign_labels(Xbar, centroids):
    N = Xbar.shape[1]             
    K = centroids.shape[1]      

    D = np.zeros((N, K)) # D[i][j] la kcach ^ 2 cua diem du lieu thu i voi tam cum thu j
    for n in range(N):
        for k in range(K):
            diff = Xbar[:, n] - centroids[:, k]      # mẫu n và tâm k đều là CỘT, shape (d+1,)
            D[n, k] = np.sum(diff * diff)

    labels = np.argmin(D, axis=1)    # shape (N,): chỉ số tâm gần nhất của từng mẫu
    return labels


def kmeans_update_centroids(Xbar, labels, K):
    d, N = Xbar.shape

    centroids = np.zeros((d, K))

    for k in range(K):
        Xk = Xbar[:, labels == k] # lay cac diem du lieu thuoc cum k

        if Xk.shape[1] == 0:
            # cụm rỗng: np.mean của mảng rỗng ra nan, nên đặt lại tâm này ở một mẫu ngẫu nhiên
            centroids[:, k] = Xbar[:, np.random.randint(N)]
        else:
            centroids[:, k] = np.mean(Xk, axis=1) # cap nhat tam cum bang trung binh cong

    return centroids


def has_converged(centroids, new_centroids):
    # ktra dieu kien dung cua thuat toan
    # mỗi tâm là một CỘT nên duyệt centroids.T (duyệt centroids sẽ lấy từng HÀNG = từng chiều)
    return (set([tuple(a) for a in centroids.T]) == set([tuple(a) for a in new_centroids.T]))


def Kmeans(Xbar, K, max_iter=1000):
    centroids = kmeans_init_random(Xbar, K)
    labels = []
    it = 0 # iteration

    while True:
        # buoc 1: gan moi diem vao cum co tam gan nhat
        labels = kmeans_assign_labels(Xbar, centroids)

        # buoc 2: cap nhat tam cum = trung binh cong cac diem trong cum
        new_centroids = kmeans_update_centroids(Xbar, labels, K)

        # dung khi tam cum khong doi nua (labels luc nay khop voi centroids)
        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids
        it += 1

        if it >= max_iter:               # chốt chặn phòng khi không hội tụ
            labels = kmeans_assign_labels(Xbar, centroids)
            break

    return centroids, labels, it


centroids, labels_train, it = Kmeans(Xbar_train, K)
print("Số vòng lặp:", it)
print("Số mẫu mỗi cụm:", np.bincount(labels_train, minlength=K))


# ================================== DANH GIA   ===========================
# Cụm chỉ mang số 0..K-1 do thuật toán tự đặt, chưa phải chữ số thật.
# Cách đánh giá: gán mỗi cụm thành chữ số chiếm đa số TRONG CỤM (dùng nhãn Y_train),
# rồi dự đoán mẫu test = chữ số của cụm có tâm gần nhất.

cluster_to_digit = np.zeros(K, dtype=int)
for k in range(K):
    digits_in_k = Y_train[labels_train == k]               # nhãn thật của các mẫu thuộc cụm k
    if len(digits_in_k) > 0:
        cluster_to_digit[k] = np.argmax(np.bincount(digits_in_k, minlength=n_classes))
print("Cụm -> chữ số:", cluster_to_digit)

y_pred_train = cluster_to_digit[labels_train]              # shape (1600,)
labels_test = kmeans_assign_labels(Xbar_test, centroids)   # cụm gần nhất của mẫu test, shape (400,)
y_pred_test = cluster_to_digit[labels_test]                # shape (400,)

print("Accuracy train:", np.mean(y_pred_train == Y_train))
print("Accuracy test :", np.mean(y_pred_test == Y_test))

# Vẽ tâm của từng cụm như một ảnh (bỏ hàng bias ở cuối)
side = int(np.sqrt(d))                                     # 28 với MNIST
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for k, ax in enumerate(axes.flatten()):
    ax.imshow(centroids[:d, k].reshape(side, side), cmap="gray")
    ax.set_title(f"cụm {k} -> {cluster_to_digit[k]}")
    ax.axis("off")
plt.show()