from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import abc
import tensorflow as tf

# different category:
from style_vae.model import Vae

# same category:
from style_vae.train.vae_trainer_config import VaeTrainerConfig


class VaeTrainer(object, metaclass=abc.ABCMeta):

    def __init__(self, model: Vae, config: VaeTrainerConfig, sess: tf.Session):
        self._model = model
        self._config = config
        self._sess = sess

    def train(self, dataset):
        raise NotImplementedError

    def validate(self, dataset):
        raise NotImplementedError

    def save(self, save_path):
        raise NotImplementedError

    def load(self, save_path):
        raise NotImplementedError
