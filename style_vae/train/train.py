# 3rd party:
import fire
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

# different category:
from style_vae.model import StyleVae, Config
from style_vae.data import Dataset
from style_vae.output import OUT

# same category:
from style_vae.train.style_vae_trainer import StyleVaeTrainer
from style_vae.train.vae_trainer_config import VaeTrainerConfig


def train(load: bool):
    trainer = _build_trainer()
    dataset = Dataset.get_cifar10()

    save_path = OUT
    if load:
        trainer.load(save_path)
    trainer.train(dataset)
    trainer.validate(dataset)
    trainer.save(save_path)


def test():
    trainer = _build_trainer()
    dataset = Dataset.get_cifar10()

    save_path = OUT
    trainer.load(save_path)
    result = trainer.test(dataset)
    for i in range(10):
        plt.figure()
        plt.imshow(np.uint8(255 * np.clip(result['recon_img'][i, :, :, 0], 0, 1)))
        plt.gray()
    plt.show()


def _build_trainer() -> StyleVaeTrainer:
    # model
    model_config = Config()
    print(model_config)
    model = StyleVae(model_config)

    # trainer
    trainer_config = VaeTrainerConfig()
    print(trainer_config)
    sess = tf.Session()
    trainer = StyleVaeTrainer(model, trainer_config, sess)

    # init
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    return trainer


if __name__ == '__main__':
    fire.Fire(train)
