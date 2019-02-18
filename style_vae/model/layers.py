# 3rd party:
import fire
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# same category:
class VaeLayers(object):
    @staticmethod
    def normalize(x, epsilon=1e-8):
        """Pixelwise feature vector normalization"""
        with tf.variable_scope('normalize'):
            x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True) + epsilon)
            return x

    @staticmethod
    def blur_2d(x):
        with tf.variable_scope('blur2d'):
            kernel = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
            kernel /= np.sum(kernel)
            blur_filter = tf.constant(kernel, dtype=tf.float32)
            blur_filter = tf.tile(blur_filter[:, :, None, None], [1, 1, int(x.shape[3]), 1])
            x = tf.nn.depthwise_conv2d(x, blur_filter, strides=[1, 1, 1, 1], padding='SAME')
            return x

    @staticmethod
    def add_bias(x):
        with tf.variable_scope('Bias'):
            b = tf.Variable(initial_value=tf.zeros(shape=(1, 1, 1, x.shape[3])), name='b')
            return x + b

    @staticmethod
    def additive_noise(x):
        """scaling noise bi-cubic to fit x and adding it"""
        with tf.variable_scope('Additive-Noise'):
            n = tf.random_normal([tf.shape(x)[0], x.shape[1], x.shape[2], 1], dtype=x.dtype)
            b = tf.Variable(initial_value=tf.zeros(shape=(1, 1, 1, x.shape[3])), name='b')
            return x + b * n

    @staticmethod
    def stylize(x, style):
        """stylize the feature maps AdaIN in paper"""
        with tf.variable_scope('Stylize'):
            style_affined = layers.Dense(units=x.shape[3] * 2, name='dense')(style)
            style_scale, style_bias = tf.split(style_affined, 2, axis=1)

        return x * (style_scale[:, None, None, :] + 1) + style_bias[:, None, None, :]

    @staticmethod
    def conv3(x, f_maps, activation, blur=False, add_noise=True):
        name = 'conv3'
        x = layers.Conv2D(filters=f_maps,
                          kernel_size=3,
                          padding='same',
                          activation=activation,
                          name=name)(x)
        if blur:
            x = VaeLayers.blur_2d(x)

        if add_noise:
            x = VaeLayers.additive_noise(x)

        x = VaeLayers.add_bias(x)
        x = activation(x)
        return x

    @staticmethod
    def conv3_stride2(x, f_maps, activation):
        name = 'conv3s2'
        x = layers.Conv2D(filters=f_maps,
                          kernel_size=3,
                          padding='same',
                          strides=2,
                          activation=activation,
                          name=name)(x)
        return x

    @staticmethod
    def cell_up(x, f_maps, style, activation=layers.LeakyReLU(0.2)):
        with tf.variable_scope('cell-up'):
            size = [2 * int(x.shape[1]), 2 * int(x.shape[2])]
            x = tf.image.resize_nearest_neighbor(x, size, align_corners=True)
            x = VaeLayers.conv3(x, f_maps, activation, blur=True)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
            x = VaeLayers.conv3(x, f_maps, activation)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
        return x

    @staticmethod
    def to_rgb(x, activation=layers.LeakyReLU(0.2)):
        name = 'to_rgb'
        x = layers.Conv2D(filters=3,
                          kernel_size=3,
                          padding='same',
                          strides=1,
                          activation=activation,
                          kernel_initializer='glorot_normal',
                          name=name)(x)
        return x

    @staticmethod
    def to_gray(x, activation=layers.LeakyReLU(0.2)):
        name = 'to_gray'
        x = layers.Conv2D(filters=1,
                          kernel_size=3,
                          padding='same',
                          strides=1,
                          activation=activation,
                          kernel_initializer='glorot_normal',
                          name=name)(x)
        return x

    @staticmethod
    def first_cell_up(var, style, f_maps, activation=layers.LeakyReLU(0.2)):
        with tf.variable_scope('first-cell-up'):
            x = VaeLayers.additive_noise(var)
            x = VaeLayers.add_bias(x)
            x = activation(x)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
            x = VaeLayers.conv3(x, f_maps, activation)
            x = VaeLayers.normalize(x)
            x = VaeLayers.stylize(x, style)
        return x

    @staticmethod
    def cell_down(x, f_maps, activation=layers.LeakyReLU(0.2)):
        x = VaeLayers.conv3(x, f_maps, activation)
        x = VaeLayers.conv3_stride2(x, f_maps - 1, activation)
        return x
