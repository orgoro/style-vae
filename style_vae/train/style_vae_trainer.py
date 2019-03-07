# 3rd party:
import tensorflow as tf
from tensorflow.keras import losses as loss
from dataclasses import dataclass
from tqdm import tqdm
from os import path

# different category:
from style_vae.output import OUT
from style_vae.model import StyleVae, PerceptualModel

# same category:
from style_vae.train.vae_trainer_config import VaeTrainerConfig


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


class StyleVaeTrainer(object):
    EPOCH = 0

    def __init__(self, model: StyleVae, config: VaeTrainerConfig, sess: tf.Session):
        self._model = model
        self._config = config
        self._sess = sess
        self._ph = None  # type: StyleVaePh
        self._stub = None  # type: StyleVaeStub
        self._global_step = tf.train.create_global_step()
        self._build_graph()
        self._saver = tf.train.Saver()
        self._graph_writer = tf.summary.FileWriter(OUT, graph=self._sess.graph)
        self._train_summary, self._val_summary = self._add_summary()

    def _build_graph(self):
        img_dim = self._model.config.img_dim
        ch = self._model.config.num_channels
        img_ph = tf.placeholder(dtype=tf.float32, shape=(None, img_dim, img_dim, ch), name='img_ph')
        self._ph = StyleVaePh(img_ph)

        code = self._model.encode(img_ph)  # tf.Variable(np.ones((128, 200), dtype=np.float32))
        # code_loss =
        recon_img = self._model.decode(code)
        recon_loss = self._build_recon_loss(img_ph, recon_img)

        optimizer = tf.train.AdamOptimizer(self._config.lr)
        opt_step = optimizer.minimize(recon_loss, global_step=self._global_step)
        self._stub = StyleVaeStub(code, recon_img, recon_loss, opt_step)

    def _build_recon_loss(self, img_ph, recon_img):
        loss_type = self._config.recon_loss
        with tf.variable_scope(f'recon-loss/{loss_type}'):
            if loss_type == 'perceptual':
                perceptual_model = PerceptualModel()
                f1 = perceptual_model(img_ph)
                f2 = perceptual_model(recon_img)
                recon_loss = tf.reduce_mean(tf.square(f1 - f2))
            elif loss_type == 'l2':
                recon_loss = tf.reduce_mean(loss.binary_crossentropy(img_ph, recon_img))
            else:
                raise NotImplementedError(f'loss type: {loss_type} not implemented')
        return recon_loss

    def _add_summary(self):
        train_summary_recon_loss = tf.summary.scalar('train/recon_loss', self._stub.recon_loss)
        train_summary_recon = tf.summary.image('train/recon', self._stub.recon_img)
        train_summary_src = tf.summary.image('train/src', self._ph.img_ph)
        train_summary = tf.summary.merge([train_summary_recon_loss, train_summary_recon, train_summary_src])

        val_summary_recon_loss = tf.summary.scalar('val/recon_loss', self._stub.recon_loss)
        val_summary_recon = tf.summary.image('val/recon', self._stub.recon_img)
        val_summary_src = tf.summary.image('val/src', self._ph.img_ph)
        val_summary = tf.summary.merge([val_summary_recon_loss, val_summary_recon, val_summary_src])

        return train_summary, val_summary

    def train(self, dataset):
        fetches = self._stub.get_train_fetch()
        for e in range(self._config.num_epochs):
            self.EPOCH = e
            self._run_epoch(dataset.train, fetches, phase='train')
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
                summary_op = self._train_summary if phase == 'train' else self._val_summary
                summary, global_step = self._sess.run([summary_op, self._global_step], self._ph.get_feed(img))
                self._graph_writer.add_summary(summary, global_step=global_step)

        avg_loss /= steps
        print(f'\n{phase} epoch {self.EPOCH:3}/{self._config.num_epochs:3} --> avg_loss: {avg_loss:.4}')

    def validate(self, dataset):
        fetches = self._stub.get_validate_fetch()
        self._run_epoch(dataset.val, fetches, phase='val')

    def save(self, save_path=OUT):
        print('save')
        global_step = self._global_step.eval(session=self._sess)
        self._saver.save(self._sess, path.join(save_path, 'model.ckpt'),
                         global_step=global_step,
                         latest_filename='latest.ckpt')

    def load(self, save_path=OUT):
        ckpt = tf.train.latest_checkpoint(save_path, 'latest.ckpt')
        if ckpt:
            print('restore')
            self._saver.restore(self._sess, ckpt)
        else:
            print('restore failed!')

    def test(self, dataset):
        img = dataset.test[0:10]
        feed_dict = self._ph.get_feed(img)
        fetch = self._stub.get_validate_fetch()
        result = self._sess.run(fetch, feed_dict)
        return result
