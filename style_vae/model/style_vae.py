from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import tensorflow as tf
from tensorflow.keras import layers
from dataclasses import dataclass
import numpy as np

# different category:


# same category:
from style_vae.model.layers import VaeLayers


@dataclass
class Config(object):
    code_size: int = 512
    img_dim: int = 64
    batch_size: int = 64
    num_channels: int = 1
    fmap_base: int = 8192
    fmap_decay: int = 1.0
    fmap_max: int = 512

    def __str__(self):
        res = 'VaeConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


class StyleVae:

    def __init__(self, config: Config):
        self.config = config

    def nf(self, res):
        conf = self.config
        return min(int(conf.fmap_base / (2.0 ** (res * conf.fmap_decay))), conf.fmap_max)

    def encode(self, image: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope('Encoder'):
            x = image
            log2_dim = int(np.log2(self.config.img_dim))

            for res in np.arange(log2_dim, 2, -1):
                x = VaeLayers.cell_down(x, self.nf(res-1))

            x = layers.Flatten()(x)
            x = layers.Dense(self.config.code_size)(x)
        return x

    def decode(self, code: tf.Tensor):
        with tf.variable_scope('Decoder'):
            # first block
            first_var = tf.Variable(initial_value=tf.random_normal((1, 4, 4, self.nf(1))))
            x = VaeLayers.first_cell_up(first_var, f_maps=self.nf(1), style=code)

            # blocks
            log2_dim = int(np.log2(self.config.img_dim))
            for res in range(3, log2_dim + 1):
                x = VaeLayers.cell_up(x, f_maps=self.nf(res-1), style=code)

            # convert to image
            if self.config.num_channels == 3:
                x = VaeLayers.to_rgb(x)
            else:
                x = VaeLayers.to_gray(x)

        return x
