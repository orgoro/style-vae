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
    code_size: int = 256
    img_dim: int = 64
    batch_size: int = 16
    num_channels: int = 3
    fmap_base: int = 8192
    fmap_decay: int = 1.0
    fmap_max: int = 512
    mapper_layers: int = 4
    discrim_layers: int = 4
    seed: int = 42

    def __str__(self):
        res = 'VaeConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


class StyleVae:

    def __init__(self, config: Config):
        self.config = config
        tf.set_random_seed(config.seed)

    def nf(self, res):
        conf = self.config
        return min(int(conf.fmap_base / (2.0 ** (res * conf.fmap_decay))), conf.fmap_max)

    def encode(self, image: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        with tf.variable_scope('Encoder'):
            log2_dim = int(np.log2(self.config.img_dim))
            x = VaeLayers.from_image(image, log2_dim)

            for res in np.arange(log2_dim, 2, -1):
                f_maps = (self.nf(res - 1), self.nf(res - 2))
                x = VaeLayers.cell_down(x, f_maps)

            x = layers.Flatten()(x)
            code_mean = layers.Dense(self.config.code_size)(x)
            code_log_std = layers.Dense(self.config.code_size)(x)
        return code_mean, code_log_std

    def decode(self, code: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope('Decoder'):
            # first block
            first_var = tf.Variable(initial_value=tf.random_normal((1, 4, 4, self.nf(1))), dtype=tf.float32)
            x = VaeLayers.first_cell_up(first_var, f_maps=self.nf(1), style=code)

            # blocks
            log2_dim = int(np.log2(self.config.img_dim))
            for res in range(3, log2_dim + 1):
                x = VaeLayers.cell_up(x, f_maps=self.nf(res - 1), style=code)

            # convert to image
            if self.config.num_channels == 3:
                x = VaeLayers.to_rgb(x)
            else:
                x = VaeLayers.to_gray(x)

        return x

    def map(self, code_mean: tf.Tensor, code_log_std: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        with tf.variable_scope('Reparam-Trick'):
            x = tf.random_normal(shape=(tf.shape(code_mean)[0], code_mean.shape[1]))
            x = x * tf.exp(code_log_std) + code_mean

        with tf.variable_scope('Mapper'):
            x_ph = tf.placeholder_with_default(x, shape=x.shape)
            x = tf.identity(x_ph)
            for l in range(self.config.mapper_layers):
                x = VaeLayers.map_cell(x)
            return x, x_ph

    def discriminate(self, img):
        with tf.variable_scope('Discriminate', reuse=tf.AUTO_REUSE):
            log2_dim = int(np.log2(self.config.img_dim))
            x = VaeLayers.from_image(img, log2_dim)

            for res in np.arange(log2_dim, 2, -1):
                f_maps = (self.nf(res - 1), self.nf(res - 2))
                x = VaeLayers.cell_down(x, f_maps)

            x = layers.Flatten()(x)
            for l in range(self.config.discrim_layers):
                x = VaeLayers.map_cell(x)

            x = layers.Dense(1)(x)
            x = tf.sigmoid(x)
            return x