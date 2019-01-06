from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import fire
import tensorflow as tf
from tensorflow import keras


# same category:
class Layers(object):

    @staticmethod
    def normalize(x, epsilon=1e-8):
        """Pixelwise feature vector normalization"""
        with tf.variable_scope('Pixel-Norm'):
            return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon, name='norm-factor')

    @staticmethod
    def additive_noise(x, noise):
        """scaling noise bi-cubic to fit x and adding it"""
        with tf.variable_scope('Additive-Noise'):
            n = tf.image.resize_bicubic(noise, size=x.shape[1:3], name='resize-noise')[None]
            return x + n

    @staticmethod
    def stylize(x, style):
        """stylize the feature maps"""
        with tf.variable_scope('Stylize'):
            style_len = tf.cast(tf.divide(tf.shape(style)[1], 2), tf.int64, name='style-len')
            scale = style[:, :style_len]
            bias = style[:, style_len:]
        return scale * x + bias

    @staticmethod
    def cell_up(x):
        pass

    @staticmethod
    def cell_down(x):
        pass


if __name__ == '__main__':
    fire.Fire()
