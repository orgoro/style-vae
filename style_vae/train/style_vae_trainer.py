from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses as loss
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
    EPOCH = 0

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
        # l2_loss = tf.reduce_mean(tf.square(img_ph - recon_img), name='recon_loss')
        recon_loss = tf.reduce_mean(loss.binary_crossentropy(img_ph, recon_img))

        optimizer = tf.train.AdamOptimizer(self._config.lr)
        opt_step = optimizer.minimize(recon_loss)
        self._stub = StyleVaeStub(code, recon_img, recon_loss, opt_step)

        # summary
        train_summary_recon_loss = tf.summary.scalar('train/recon_loss', recon_loss)
        train_summary_recon = tf.summary.image('train/recon', recon_img)
        train_summary_src = tf.summary.image('train/src', img_ph)
        self._train_summary = tf.summary.merge([train_summary_recon_loss, train_summary_recon, train_summary_src])

        val_summary_recon_loss = tf.summary.scalar('val/recon_loss', recon_loss)
        val_summary_recon = tf.summary.scalar('val/recon', recon_img)
        val_summary_src = tf.summary.image('train/src', img_ph)
        self._val_summary = tf.summary.merge([val_summary_recon_loss, val_summary_recon, val_summary_src])

    def train(self, dataset):
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        phase = 'train'

        fetches = self._stub.get_train_fetch()
        for e in range(self._config.num_epochs):
            self.EPOCH = e
            self._run_epoch(dataset.train, fetches, phase)
            self.validate(dataset)

    def _run_epoch(self, images, fetches, phase):
        steps = images.shape[0] // self._config.batch_size
        progress = tqdm(range(steps))
        avg_loss = 0
        for step in progress:
            offset = step * self._config.batch_size
            img = images[offset: offset + self._config.batch_size]
            res = self._sess.run(fetches, self._ph.get_feed(img))
            cur_loss = res['recon_loss']
            progress.set_description(f'{phase} epoch {self.EPOCH:3}/{self._config.num_epochs:3}|{step:4}/{steps:4}'
                                     f' | recon_loss: {cur_loss:.4}')
            avg_loss += cur_loss

            if step == len(progress) - 1:
                summary = self._train_summary if phase == 'train' else self._val_summary
                _ = self._sess.run(summary, self._ph.get_feed(img))
                examples = [(img[i], res['recon_img'][i]) for i in range(self._config.batch_size)]
                pairs = [np.concatenate(example, axis=0) for example in examples[:10]]
                res = np.concatenate(pairs, axis=1)
                plt.figure(figsize=(10, 4))
                plt.imshow(res, aspect='equal')
                plt.axis('off'), plt.suptitle(f'{phase}-e{self.EPOCH}-l{avg_loss / steps}')
                plt.savefig(path.join(OUT, f'{phase}-recon.png'))

        avg_loss /= steps
        print(f'\n{phase} epoch {self.EPOCH:3}/{self._config.num_epochs:3} --> avg_loss: {avg_loss:.4}')

    def validate(self, dataset):
        phase = 'val'
        fetches = self._stub.get_validate_fetch()
        self._run_epoch(dataset.val, fetches, phase)

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
