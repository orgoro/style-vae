from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from tensorflow.keras import datasets
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Dataset:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __str__(self):
        res = 'Dataset:\n'
        for k, v in vars(self).items():
            val = v if isinstance(v, str) else (v.shape, v.dtype)
            res += f'o {k:10}|{val}\n'

        return res

    @classmethod
    def get_cifar10(cls):
        cifar10 = datasets.cifar10.load_data()
        (x_train, _), (x_test, _) = cifar10
        num_val = x_train.shape[0] // 10
        return Dataset('cifar10',
                       np.float32(x_train[num_val:] / 255.),
                       np.float32(x_train[:num_val] / 255.),
                       np.float32(x_test / 255.))

    @classmethod
    def get_mnist64(cls):
        mnist = datasets.mnist.load_data()
        (x_train, _), (x_test, _) = mnist
        train_size = x_train.shape[0]
        train_64 = np.array([cv2.resize(x_train[i], (64, 64)) for i in range(train_size)])
        test_size = x_test.shape[0]
        test_64 = np.array([cv2.resize(x_test[i], (64, 64)) for i in range(test_size)])
        num_val = x_train.shape[0] // 10
        return Dataset('mnist64',
                       np.float32(train_64[num_val:] / 255.),
                       np.float32(train_64[:num_val] / 255.),
                       np.float32(test_64 / 255.))


if __name__ == '__main__':
    mnist64 = Dataset.get_mnist64()
    print(mnist64)
    from matplotlib import pyplot as plt

    plt.figure(), plt.gray(), plt.imshow(np.uint8(mnist64.train[0,:,:] * 255)), plt.colormaps('gray'),plt.show()
