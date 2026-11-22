import gzip
import os
import urllib.request

import numpy as np

MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]
IMAGES = 'train-images-idx3-ubyte.gz'
LABELS = 'train-labels-idx1-ubyte.gz'


def _download(name, folder):
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path
    for base in MIRRORS:
        try:
            print('Đang tải', base + name, '...')
            # Tải vào file .part, xong hẳn mới đổi tên:
            # bấm dừng giữa chừng cũng không để lại file hỏng.
            urllib.request.urlretrieve(base + name, path + '.part')
            os.replace(path + '.part', path)
            return path
        except Exception as e:
            print('  Lỗi:', e)
    raise RuntimeError('Không tải được ' + name + ' từ các nguồn đã thử. Kiểm tra kết nối mạng.')


def load_mnist(folder='mnist_data'):
    """Trả về (X, y): X có shape (60000, 784) kiểu uint8, y có shape (60000,)."""
    os.makedirs(folder, exist_ok=True)

    with gzip.open(_download(IMAGES, folder)) as f:
        buf = f.read()
    n, rows, cols = [int(v) for v in np.frombuffer(buf, dtype='>i4', count=3, offset=4)]
    X = np.frombuffer(buf, dtype=np.uint8, offset=16).reshape(n, rows * cols)

    with gzip.open(_download(LABELS, folder)) as f:
        buf = f.read()
    y = np.frombuffer(buf, dtype=np.uint8, offset=8)

    return X, y


if __name__ == '__main__':
    X, y = load_mnist()
    print('X:', X.shape, 'y:', y.shape, 'nhãn đầu:', y[:10])