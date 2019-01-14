from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from tensorflow.keras import datasets
from dataclasses import dataclass
import numpy as np


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


if __name__ == '__main__':
    cifar10 = Dataset.get_cifar10()
    print(cifar10)
