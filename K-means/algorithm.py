from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from mnist_loader import load_mnist

np.random.seed(18)

# Tóm tắt thuật toán K-means:
#   1. Chọn K tâm ban đầu.
#   2. Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất.
#   3. Tâm mới của mỗi cụm = trung bình cộng các điểm trong cụm.
#   4. Lặp lại bước 2 và 3 đến khi tâm không đổi nữa (hội tụ).
#   Hàm mất mát J = tổng bình phương khoảng cách từ mỗi điểm đến tâm của nó.
#   J không cần tính để cập nhật tâm, chỉ dùng để theo dõi: nó giảm dần qua các vòng lặp.

# ---------------------------------------------------------------
# 1. Tải MNIST (lần đầu sẽ tải ~10MB vào thư mục mnist_data)
# ---------------------------------------------------------------
data, target = load_mnist()

K = 10       # số cụm (10 chữ số 0-9)
N = 10000    # số ảnh lấy ngẫu nhiên

# Lưu lại idx để lấy được nhãn thật y (chỉ dùng để đánh giá ở bước 5)
idx = np.random.choice(data.shape[0], N, replace=False)
X = data[idx] / 255.0                 # đưa pixel về [0, 1]
y = target[idx].astype(int)

# ---------------------------------------------------------------
# 2. Phân cụm
# ---------------------------------------------------------------
K_means = KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
pred_label = K_means.predict(X)

# ---------------------------------------------------------------
# 3. Hiển thị 10 tâm cụm
# ---------------------------------------------------------------
centers = K_means.cluster_centers_          # shape (10, 784)

plt.figure(figsize=(10, 4))
for k in range(K):
    plt.subplot(2, 5, k + 1)
    plt.imshow(centers[k].reshape(28, 28), cmap='gray')
    plt.title('Cụm %d' % k)
    plt.axis('off')
plt.tight_layout()

# ---------------------------------------------------------------
# 4. Với mỗi cụm, lấy 20 ảnh thật gần tâm nhất
#    (cột đầu: tâm cụm, các cột sau: ảnh thật)
# ---------------------------------------------------------------
n_show = 20
fig, axes = plt.subplots(K, n_show + 1, figsize=(n_show + 1, K))
for ax in axes.ravel():
    ax.axis('off')

for k in range(K):
    Xk = X[pred_label == k]                       # các ảnh thuộc cụm k
    nn = NearestNeighbors(n_neighbors=min(n_show, len(Xk))).fit(Xk)
    _, ind = nn.kneighbors(centers[k].reshape(1, -1))

    axes[k, 0].imshow(centers[k].reshape(28, 28), cmap='gray')
    for j, i in enumerate(ind[0]):
        axes[k, j + 1].imshow(Xk[i].reshape(28, 28), cmap='gray')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5. Đánh giá: gán mỗi cụm cho chữ số xuất hiện nhiều nhất trong cụm
# ---------------------------------------------------------------
cluster_to_digit = {}
for k in range(K):
    digits = y[pred_label == k]
    cluster_to_digit[k] = np.bincount(digits, minlength=10).argmax()
    print('Cụm %d -> chữ số %d (%d ảnh)' % (k, cluster_to_digit[k], len(digits)))

mapped = np.array([cluster_to_digit[k] for k in pred_label])
print('Độ chính xác: %.2f%%' % (100 * np.mean(mapped == y)))