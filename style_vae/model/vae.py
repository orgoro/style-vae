from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import fire
import abc
import tensorflow as tf


# same category:


# different category:


class VaeConfig(object):
    def __init__(self, code_size: int, img_dim: int, batch_size: int):
        self.code_size = code_size
        self.img_dim = img_dim
        self.batch_size = batch_size


class Vae(object, metaclass=abc.ABCMeta):
    """ base class for variational auto encoder """

    def __init__(self, vae_config: VaeConfig):
        self.config = vae_config

    def encode(self, image: tf.Tensor) -> tf.Tensor:
        """
        encode a batch of images to code vectors
        :param image: tensor tf.float32 BxHxWxC [0,1] 
        :return: code vector tensor tf.float32 BxD
        """
        raise NotImplementedError

    def decode(self, code: tf.Tensor) -> tf.Tensor:
        """
        dencode a batch of code vectors to images
        :param code: tensor tf.float32 BxH
        :return: images vector tensor tf.float32 BxHxWxC
        """
        raise NotImplementedError
