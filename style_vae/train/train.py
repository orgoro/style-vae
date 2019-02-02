from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import fire
import tensorflow as tf

# different category:
from style_vae.model import StyleVae, VaeConfig
from style_vae.data import Dataset
from style_vae.output import OUT

# same category:
from style_vae.train.style_vae_trainer import StyleVaeTrainer
from style_vae.train.vae_trainer import VaeTrainerConfig


def train():
    trainer = _build_trainer()
    dataset = Dataset.get_mnist64()

    save_path = OUT
    trainer.load(save_path)
    trainer.train(dataset)
    trainer.validate(dataset)
    trainer.save(save_path)


def _build_trainer() -> StyleVaeTrainer:
    model_config = VaeConfig()
    print(model_config)

    model = StyleVae(model_config)
    trainer_config = VaeTrainerConfig()
    print(trainer_config)

    sess = tf.Session()
    trainer = StyleVaeTrainer(model, trainer_config, sess)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    return trainer


if __name__ == '__main__':
    fire.Fire(train)
