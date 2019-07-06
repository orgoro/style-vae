# 3rd party:
from dataclasses import dataclass
import fire
import tensorflow as tf
from os import path

# different category:
from style_vae.model import StyleVae, Config

# same category:
from style_vae.train.style_vae_trainer import StyleVaeTrainer


@dataclass
class VaeTrainerConfig:
    name: str = 'default-vae'
    reload_model: bool = True
    save_model: bool = True
    batch_size: int = 16
    num_epochs: int = 20
    lr: float = 2e-5
    recon_loss: str = 'perceptual'
    latent_weight: float = 2.
    adv_weight: float = 1e1
    data_regex: str = path.join('/data', 'ffhq_256', '*.png')

    def __str__(self):
        res = 'VaeTrainerConfig:\n'
        for k, v in vars(self).items():
            res += f'o {k:15}|{v}\n'
        return res


def train(load: bool):
    trainer = _build_trainer()
    if load:
        trainer.load()
    trainer.train()


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
