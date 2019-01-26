from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import tensorflow as tf
from tensorflow.keras import layers

# different category:


# same category:
from style_vae.model.layers import VaeLayers
from style_vae.model.vae import Vae


class StyleVae(Vae):

    def encode(self, image: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope('Encoder'):
            x = image
            while x.shape[1] > 4:
                x = VaeLayers.cell_down(x, self.config.code_size)
            x = layers.Flatten()(x)
            x = layers.Dense(self.config.code_size)(x)
        return x

    def decode(self, code: tf.Tensor):
        with tf.variable_scope('Decoder'):
            noise = tf.random_normal((self.config.batch_size, self.config.img_dim, self.config.img_dim, 1))
            first_var = tf.Variable(initial_value=tf.random_normal((1, 4, 4, self.config.code_size // 2)))
            x = VaeLayers.first_cell_up(first_var, f_maps=self.config.code_size // 2, noise=noise, style=code)

            while x.shape[1] < self.config.img_dim:
                x = VaeLayers.cell_up(x, f_maps=self.config.code_size // 2, noise=noise, style=code)
            x = VaeLayers.to_rgb(x)

        return x
