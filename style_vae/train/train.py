# 3rd party:
import fire
import tensorflow as tf

# different category:
from style_vae.model import StyleVae, Config

# same category:
from style_vae.train.style_vae_trainer import StyleVaeTrainer, VaeTrainerConfig


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
