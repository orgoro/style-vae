from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from matplotlib import pyplot as plt
import tensorflow as tf
from dataclasses import dataclass
from tqdm import tqdm

# different category:
from style_vae.model import Vae

# same category:
from style_vae.train.vae_trainer_config import VaeTrainerConfig
from style_vae.train.vae_trainer import VaeTrainer


@dataclass
class StyleVaePh:
    img_ph: tf.placeholder

    def get_feed(self, img):
        return {self.img_ph: img}


@dataclass
class StyleVaeStub:
    code: tf.Tensor
    # code_loss: tf.Tensor
    recon_img: tf.Tensor
    recon_loss: tf.Tensor
    opt_step: tf.Operation

    def get_validate_fetch(self) -> dict:
        return {'code': self.code, 'recon_img': self.recon_img, 'recon_loss': self.recon_loss}

    def get_train_fetch(self) -> dict:
        return {**{'opt_step': self.opt_step}, **self.get_validate_fetch()}


class StyleVaeTrainer(VaeTrainer):

    def __init__(self, model: Vae, config: VaeTrainerConfig, sess: tf.Session):
        super(StyleVaeTrainer, self).__init__(model, config, sess)
        self._ph = None  # type: StyleVaePh
        self._stub = None  # type: StyleVaeStub
        self._build_graph()

    def _build_graph(self):
        img_dim = self._model.config.img_dim
        img_ph = tf.placeholder(dtype=tf.float32, shape=(None, img_dim, img_dim, 3))
        self._ph = StyleVaePh(img_ph)

        code = self._model.encode(img_ph)
        # code_loss =
        recon_img = self._model.decode(code)
        recon_loss = tf.reduce_mean(tf.square(img_ph - recon_img))

        optimizer = tf.train.AdamOptimizer()
        opt_step = optimizer.minimize(recon_loss)
        self._stub = StyleVaeStub(code, recon_img, recon_loss, opt_step)

    def train(self, dataset):
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        steps = dataset.train.shape[0] // self._config.batch_size
        fetches = self._stub.get_train_fetch()
        for e in range(self._config.num_epochs):
            progress = tqdm(range(steps))
            avg_loss = 0
            for step in progress:
                offset = step * self._config.batch_size
                img = dataset.train[offset: offset + self._config.batch_size]
                res = self._sess.run(fetches, self._ph.get_feed(img))
                loss = res['recon_loss']
                progress.set_description(
                    f'epoch {e:3}/{self._config.num_epochs:3}|{step:4}/{steps:4} | recon_loss: {loss:.4}')
                avg_loss += loss

            avg_loss /= steps
            progress.set_description(f'epoch {e:3}/{self._config.num_epochs:3} --> avg_loss: {avg_loss:.4}')
            self.validate(dataset)

    def validate(self, dataset):
        steps = dataset.val.shape[0] // self._config.batch_size
        examples = []
        for e in range(self._config.num_epochs):
            progress = tqdm(range(steps))
            fetches = self._stub.get_validate_fetch()
            avg_loss = 0
            for step in progress:
                offset = step * self._config.batch_size
                img = dataset.val[offset: offset + self._config.batch_size]
                res = self._sess.run(fetches, self._ph.get_feed(img))
                loss = res['recon_loss']
                progress.set_description(
                    f'VAL epoch {e:3}/{self._config.num_epochs:3}|{step:4}/{steps:4} | recon_loss: {loss:.4}')
                avg_loss += loss

                if step == 0:
                    examples = [(img[i], res['recon_img'][i]) for i in range(self._config.batch_size)]

            avg_loss /= steps
            progress.set_description(f'VAL epoch {e:3}/{self._config.num_epochs:3} | avg_loss: {avg_loss:.4}')

            cols = 10
            fig, ax = plt.subplots(2, cols)
            for i in range(cols):
                for k in [0, 1]:
                    cur_ax = ax[k, i]
                    cur_ax.imshow(examples[i][k])
                    cur_ax.set_axis_off()

            plt.tight_layout(), plt.show()

    def save(self, save_path):
        raise NotImplementedError

    def load(self, save_path):
        raise NotImplementedError
