from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import fire
import tensorflow as tf
from tensorflow.keras import layers


# same category:
class VaeLayers(object):
    @staticmethod
    def normalize(x, epsilon=1e-8):
        """Pixelwise feature vector normalization"""
        with tf.variable_scope('normalize'):
            # return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon, name='norm-factor')
            mean, var = tf.nn.moments(x, axes=3)
            std = tf.sqrt(var + epsilon)
            x = (x - mean[:, :, :, None]) / (std[:, :, :, None] + epsilon)
            return x

    @staticmethod
    def additive_noise(x, noise):
        """scaling noise bi-cubic to fit x and adding it"""
        with tf.variable_scope('Additive-Noise'):
            n = tf.image.resize_bicubic(noise, size=x.shape[1:3], name='resize-noise')
            b = tf.Variable(initial_value=tf.ones(shape=(1, 1, 1, tf.shape(x)[-1])), name='b')
            return x + b * n

    @staticmethod
    def stylize(x, style):
        """stylize the feature maps AdaIN in paper"""
        with tf.variable_scope('Stylize'):
            style_len = style.shape[1]
            a_scale = tf.Variable(initial_value=tf.ones(shape=(1, style_len)), name='a-scale')
            a_bias = tf.Variable(initial_value=tf.ones(shape=(1, style_len)), name='a-bias')
            style_affine = style * a_scale + a_bias
            scale = style_affine[:, None, None, :style_len // 2]
            bias = style_affine[:, None, None, style_len // 2:]
        return scale * x + bias

    @staticmethod
    def conv3(x, f_maps, activation):
        x = layers.Conv2D(filters=f_maps, kernel_size=3, padding='same', activation=activation)(x)
        return x

    @staticmethod
    def conv3_stride2(x, f_maps, activation):
        x = layers.Conv2D(filters=f_maps, kernel_size=3, padding='same', strides=2, activation=activation)(x)
        return x

    @staticmethod
    def cell_up(x, f_maps, noise, style, activation='relu'):
        with tf.variable_scope('cell-up'):
            x = layers.UpSampling2D(2)(x)
            x = VaeLayers.conv3(x, f_maps, activation)
            x = VaeLayers.additive_noise(x, noise)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
            x = VaeLayers.conv3(x, f_maps, activation)
        return x

    @staticmethod
    def first_cell_up(var, f_maps, noise, style, activation='relu'):
        with tf.variable_scope('first-cell-up'):
            x = VaeLayers.additive_noise(var, noise)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
            x = VaeLayers.conv3(x, f_maps, activation)
            x = VaeLayers.additive_noise(x, noise)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
        return x

    @staticmethod
    def cell_down(x, f_maps, activation='relu'):
        x = VaeLayers.conv3_stride2(x, f_maps, activation)
        return x
