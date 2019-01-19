from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from matplotlib import pyplot as plt
import tensorflow as tf
from dataclasses import dataclass
from tqdm import tqdm
from os import path
import numpy as np

# different category:
from style_vae.model import Vae
from style_vae.output import OUT

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
        self._saver = tf.train.Saver()
        self._graph_writer = tf.summary.FileWriter(OUT, graph=self._sess.graph)

    def _build_graph(self):
        img_dim = self._model.config.img_dim
        img_ph = tf.placeholder(dtype=tf.float32, shape=(None, img_dim, img_dim, 3), name='img_ph')
        self._ph = StyleVaePh(img_ph)

        code = self._model.encode(img_ph)  # tf.Variable(np.ones((128, 200), dtype=np.float32))
        # code_loss =
        recon_img = self._model.decode(code)
        recon_loss = tf.reduce_mean(tf.square(img_ph - recon_img), name='recon_loss')

        optimizer = tf.train.AdamOptimizer(self._config.lr)
        opt_step = optimizer.minimize(recon_loss)
        self._stub = StyleVaeStub(code, recon_img, recon_loss, opt_step)

    def train(self, dataset):
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        steps = 10  # dataset.train.shape[0] // self._config.batch_size
        fetches = self._stub.get_train_fetch()
        for e in range(self._config.num_epochs):
            # self.save()
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

                # if step == 0:
                #     examples = [(img[i], res['recon_img'][i]) for i in range(self._config.batch_size)]
                #     pairs = [np.concatenate(example, axis=0) for example in examples[:10]]
                #     res = np.concatenate(pairs, axis=1)
                #     plt.figure(figsize=(10, 4))
                #     plt.imshow(res, aspect='equal')
                #     plt.axis('off'), plt.suptitle('train')
                #     plt.show()

            avg_loss /= steps
            print(f'\nepoch {e:3}/{self._config.num_epochs:3} --> avg_loss: {avg_loss:.4}')
            # self.validate(dataset)

    def validate(self, dataset):
        steps = dataset.val.shape[0] // self._config.batch_size
        progress = tqdm(range(steps))
        fetches = self._stub.get_validate_fetch()
        avg_loss = 0
        for step in progress:
            offset = step * self._config.batch_size
            img = dataset.val[offset: offset + self._config.batch_size]
            res = self._sess.run(fetches, self._ph.get_feed(img))
            loss = res['recon_loss']
            progress.set_description(f'VAL {step:4}/{steps:4} | recon_loss: {loss:.4}')
            avg_loss += loss

            if step == 0:
                examples = [(img[i], res['recon_img'][i]) for i in range(self._config.batch_size)]
                pairs = [np.concatenate(example, axis=0) for example in examples[:10]]
                res = np.concatenate(pairs, axis=1)
                plt.figure(figsize=(10, 4))
                plt.imshow(res, aspect="equal")
                plt.axis('off'), plt.suptitle('val')
                plt.savefig(path.join(OUT, 'recon.png'))

        avg_loss /= steps
        print(f'\nVAL avg_loss: {avg_loss:.4}')

    def save(self, save_path=OUT):
        print('save')
        self._saver.save(self._sess, path.join(save_path, 'model.ckpt'))
        # tf.saved_model.simple_save(self._sess, path.join(save_path, 'model_dir'),
        #                            inputs={'img': self._ph.img_ph},
        #                            outputs={'recon_img': self._stub.recon_img})

    def load(self, save_path=OUT):
        ckpt = tf.train.latest_checkpoint(save_path)
        if ckpt:
            print('restore')
            self._sess.run(tf.global_variables_initializer())
            self._saver.restore(self._sess, ckpt)
        else:
            print('restore failed!')
