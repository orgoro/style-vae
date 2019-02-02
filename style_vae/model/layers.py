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
            mean, var = tf.nn.moments(x, axes=[1, 2])
            rstd = tf.rsqrt(var + epsilon)
            x = (x - mean[:, None, None, :]) * rstd[:, None, None, :]
            return x

    @staticmethod
    def additive_noise(x, noise):
        """scaling noise bi-cubic to fit x and adding it"""
        with tf.variable_scope('Additive-Noise'):
            n = tf.image.resize_bicubic(noise, size=x.shape[1:3], name='resize-noise')
            b = tf.Variable(initial_value=tf.zeros(shape=(1, 1, 1, x.shape[3])), name='b')
            return x + b * n

    @staticmethod
    def stylize(x, style):
        """stylize the feature maps AdaIN in paper"""
        with tf.variable_scope('Stylize'):
            style_len = style.shape[1] // 2
            style_scale, style_bias = tf.split(style, 2, axis=1)

            as_scale = tf.Variable(initial_value=tf.zeros((1, style_len)), name='a-scale-ys')
            as_bias = tf.Variable(initial_value=tf.ones((1, style_len)), name='a-bias-ys')

            ab_scale = tf.Variable(initial_value=tf.zeros((1, style_len)), name='a-scale-yb')
            ab_bias = tf.Variable(initial_value=tf.zeros((1, style_len)), name='a-bias-yb')

            scale = tf.identity(style_scale * as_scale + as_bias, name='style-scale')
            bias = tf.identity(style_bias * ab_scale + ab_bias, name='style-bias')

        return scale[:, None, None, :] * x + bias[:, None, None, :]

    @staticmethod
    def conv3(x, f_maps, activation):
        name = 'conv3'
        x = layers.Conv2D(filters=f_maps, kernel_size=3, padding='same', activation=activation, name=name)(x)
        return x

    @staticmethod
    def conv3_stride2(x, f_maps, activation):
        name = 'conv3s2'
        x = layers.Conv2D(filters=f_maps, kernel_size=3, padding='same', strides=2, activation=activation, name=name)(x)
        return x

    @staticmethod
    def cell_up(x, f_maps, noise, style, activation=layers.LeakyReLU(0.2)):
        with tf.variable_scope('cell-up'):
            x = tf.image.resize_bilinear(x, tf.shape(x)[1:3] * 2, align_corners=True)
            x = VaeLayers.conv3(x, f_maps, activation)
            x = VaeLayers.additive_noise(x, noise)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
            x = VaeLayers.conv3(x, f_maps, activation)
        return x

    @staticmethod
    def to_rgb(x, activation=layers.LeakyReLU(0.2)):
        name = 'to_rgb'
        x = layers.Conv2D(filters=3, kernel_size=3, padding='same', strides=1, activation=activation, name=name)(x)
        return x

    @staticmethod
    def to_gray(x, activation=layers.LeakyReLU(0.2)):
        name = 'to_gray'
        x = layers.Conv2D(filters=1, kernel_size=3, padding='same', strides=1, activation=activation, name=name)(x)
        return x

    @staticmethod
    def first_cell_up(var, f_maps, noise, style, activation=layers.LeakyReLU(0.2)):
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
