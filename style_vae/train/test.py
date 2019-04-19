from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import fire
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from os import path

# different category:
from style_vae.model import StyleVae, Config
from style_vae.train_output import OUT

# same category:
from style_vae.train.style_vae_trainer import StyleVaeTrainer
from style_vae.train.vae_trainer_config import VaeTrainerConfig


def plot_manifold():
    seed = np.random.randint(0, 100)
    print(f'seed - {seed}')
    np.random.seed(seed)

    gs = 9
    trainer, sess = _build_trainer(grid_size=gs)
    model = trainer.get_model()
    code_size = model.config.code_size
    img_dim = model.config.img_dim

    manifold = _generate_manifold_random(code_size, gs)
    manifold_shape = manifold.shape

    manifold_img = np.empty((gs, gs, img_dim, img_dim, 3))
    code_ph, decoded = trainer.get_decode_stubs()
    for i in range(gs):
        code = manifold[i, :, :]
        manifold_img[i] = decoded.eval(session=sess, feed_dict={code_ph: code})
    img = np.empty((gs * img_dim, gs * img_dim, 3))
    for i in range(manifold_shape[0]):
        for j in range(manifold_shape[1]):
            offset_x = i * img_dim
            offset_y = j * img_dim
            img[offset_x: offset_x+img_dim, offset_y: offset_y + img_dim] = manifold_img[i, j]

    plt.figure(), plt.imshow(img), plt.title('Manifold'), plt.axis('off')
    plt.savefig(path.join(OUT, f'grid-{seed}.jpg'), dpi=500)
    plt.show()


def _generate_manifold(code_size: int, grid_size: int) -> np.ndarray:
    vecs = np.random.randn(2, code_size) + 1
    std = 0.25
    manifold = np.zeros((grid_size, grid_size, code_size))
    s = np.linspace(-std, std, grid_size, endpoint=True)
    xx, yy = np.meshgrid(s, s)
    for i in range(grid_size):
        for j in range(grid_size):
            manifold[i, j] = xx[i, j] * vecs[0] + yy[i, j] * vecs[1]

    return manifold

def _generate_manifold_random(code_size: int, grid_size: int) -> np.ndarray:
    manifold = np.zeros((grid_size, grid_size, code_size))
    for i in range(grid_size):
        for j in range(grid_size):
            manifold[i, j] = np.random.randn(1, code_size)
    return manifold

def _build_trainer(reload_model=True, grid_size=11) -> (StyleVaeTrainer, tf.Session):
    # model
    model_config = Config()
    model_config.batch_size = grid_size

    print(model_config)
    model = StyleVae(model_config)

    # trainer
    trainer_config = VaeTrainerConfig()
    trainer_config.batch_size = grid_size
    trainer_config.reload_model = reload_model
    print(trainer_config)

    sess = tf.Session()
    trainer = StyleVaeTrainer(model, trainer_config, sess)
    if reload_model:
        trainer.load()
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
    return trainer, sess


if __name__ == '__main__':
    fire.Fire()
