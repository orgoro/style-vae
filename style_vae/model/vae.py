from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from dataclasses import dataclass
import abc
import tensorflow as tf


# different category:

# same category:


@dataclass
class VaeConfig(object):
    code_size: int = 200
    img_dim: int = 32
    batch_size: int = 128

    def __str__(self):
        res = 'VaeConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


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
