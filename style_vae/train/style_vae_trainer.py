# 3rd party:
import tensorflow as tf
from tensorflow.keras import losses as loss
from tensorflow import data
from dataclasses import dataclass
from tqdm import tqdm
from os import path
import yaml
import glob

# different category:
from style_vae.train_output import OUT
from style_vae.model import StyleVae, PerceptualModel

# same category:
from style_vae.train.vae_trainer_config import VaeTrainerConfig


@dataclass
class StyleVaePh:
    img_ph: tf.placeholder

    def get_feed(self, img):
        return {self.img_ph: img}


@dataclass
class StyleVaeSummary:
    val_loss_summary: tf.Tensor
    val_images_summary: tf.Tensor
    train_loss_summary: tf.Tensor
    train_images_summary: tf.Tensor

    def get_loss_sum(self, phase):
        if phase == 'val':
            return self.val_loss_summary
        else:
            return self.train_loss_summary

    def get_img_sum(self, phase):
        if phase == 'val':
            return self.val_images_summary
        else:
            return self.train_images_summary


@dataclass
class StyleVaeStub:
    code: tf.Tensor
    recon_img: tf.Tensor
    recon_loss: tf.Tensor
    latent_loss: tf.Tensor
    adv_loss: tf.Tensor
    combined_loss: tf.Tensor
    opt_step_g: tf.Operation
    opt_step_d: tf.Operation
    rand_img: tf.Tensor

    def get_validate_fetch(self) -> dict:
        return {'code': self.code,
                'recon_img': self.recon_img,
                'rand_img': self.rand_img,
                'recon_loss': self.recon_loss,
                'latent_loss': self.latent_loss,
                'adv_loss': self.adv_loss,
                'comb_loss': self.combined_loss}

    def get_train_fetch(self) -> dict:
        return {**{'opt_step_g': self.opt_step_g, 'opt_step_d': self.opt_step_d}, **self.get_validate_fetch()}


class StyleVaeTrainer(object):
    EPOCH = 0

    def __init__(self, model: StyleVae, config: VaeTrainerConfig, sess: tf.Session):
        self._model = model
        self._config = config
        self._sess = sess
        self._global_step = tf.train.create_global_step()

        self._img_iter_init = None
        self._num_train = None
        self._num_val = None
        self._num_test = None
        self._ph = None  # type: StyleVaePh
        self._stub = None  # type: StyleVaeStub
        self._build_graph()

        self._saver = tf.train.Saver(max_to_keep=2)
        self._graph_writer = tf.summary.FileWriter(OUT, graph=self._sess.graph)
        self._summ = None  # type: StyleVaeSummary
        self._add_summary()
        with open(path.join(OUT, 'config.yaml'), 'w') as f:
            yaml.dump([self._config, self._model.config], f, default_flow_style=False)

    def _build_graph(self):
        img_ph = self._build_ph()

        with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
            code_mean, code_log_std = self._model.encode(img_ph)

            zeros = tf.zeros_like(code_mean)
            code_mean_pad = tf.concat([code_mean, zeros], 0)
            code_log_std_pad = tf.concat([code_log_std, zeros], 0)
            code, code_ph = self._model.map(code_mean_pad, code_log_std_pad)

            decoded = self._model.decode(code)
            recon_img, rand_img = tf.split(decoded, 2)
            
        with tf.variable_scope('Discriminator'):
            fake = self._model.discriminate(rand_img)
            real = self._model.discriminate(img_ph)

        with tf.variable_scope('Loss'):
            latent_loss = self._build_latent_loss(code_mean, code_log_std)
            recon_loss = self._build_recon_loss(img_ph, recon_img)
            adv_loss = tf.reduce_sum(real - fake)
            trick_loss = tf.reduce_sum(fake)
            combined_loss = latent_loss + recon_loss + trick_loss

        vars_d = tf.trainable_variables('Discriminator')
        vars_g = tf.trainable_variables('Generator')

        with tf.variable_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self._config.lr)
            opt_step_g = optimizer.minimize(combined_loss, var_list=vars_g, global_step=self._global_step)
            with tf.control_dependencies([opt_step_g]):
                opt_step_d = optimizer.minimize(adv_loss, var_list=vars_d)

        self._stub = StyleVaeStub(code,
                                  recon_img,
                                  recon_loss,
                                  latent_loss,
                                  adv_loss,
                                  combined_loss,
                                  opt_step_g,
                                  opt_step_d,
                                  rand_img)
        self._code_ph = code_ph
        self._decoded = decoded

    def _build_ph(self):
        with tf.variable_scope('Input'):
            img_paths = glob.glob(self._config.data_regex)
            num_imgs = len(img_paths)
            num_batches = num_imgs // self._config.batch_size
            self._num_val = num_batches // 10
            self._num_test = num_batches // 10
            self._num_train = num_batches - self._num_val - self._num_test

            # lambda
            def preprocess_img(img_path):
                im = tf.read_file(img_path)
                im = tf.image.decode_image(im, channels=3)
                im = im[20:-20, :, :]
                im /= 255
                return im

            # dataset
            ds = data.Dataset.from_tensor_slices(img_paths)
            ds = ds.map(preprocess_img)
            ds = ds.batch(self._config.batch_size, drop_remainder=True)
            img_iter = ds.make_initializable_iterator()
            img_ph = img_iter.get_next('img')

            # resize image
            im_size = (self._model.config.img_dim, self._model.config.img_dim)
            img_ph = tf.image.resize_bilinear(img_ph, im_size)
            img_ph = tf.reshape(img_ph, (-1, self._model.config.img_dim, self._model.config.img_dim, 3))
            self._img_ph = img_ph
            self._iter_init = img_iter.initializer

        return img_ph

    def _build_recon_loss(self, img_ph, recon_img):
        loss_type = self._config.recon_loss
        with tf.variable_scope(f'Recon-Loss/{loss_type}'):
            if loss_type == 'perceptual':
                perceptual_model = PerceptualModel()
                f1 = perceptual_model(img_ph)
                f2 = perceptual_model(recon_img)
                recon_loss = 0.
                for ff1, ff2 in zip(f1, f2):
                    recon_loss += tf.reduce_mean(tf.reduce_sum(tf.square(ff1 - ff2), axis=1))
            elif loss_type == 'l2':
                recon_loss = tf.reduce_mean(loss.binary_crossentropy(img_ph, recon_img))
            else:
                raise NotImplementedError(f'loss type: {loss_type} not implemented')
        return recon_loss

    def _build_latent_loss(self, code_mean, code_log_std):
        with tf.variable_scope(f'Latent-Loss'):
            kl_loss = -0.5 * tf.reduce_sum(1 + 2 * code_log_std - tf.square(code_mean) - tf.exp(2 * code_log_std), 1)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= self._config.latent_weight
        return kl_loss

    def _add_summary(self):
        with tf.variable_scope('Summary'):
            ts_comb_loss = tf.summary.scalar('train/comb_loss', self._stub.combined_loss)
            ts_latent_loss = tf.summary.scalar('train/latent_loss', self._stub.latent_loss)
            ts_adv_loss = tf.summary.scalar('train/adv_loss', self._stub.adv_loss)
            ts_recon_loss = tf.summary.scalar('train/recon_loss', self._stub.recon_loss)
            ts_recon = tf.summary.image('train/recon', tf.clip_by_value(self._stub.recon_img, 0, 1))
            ts_src = tf.summary.image('train/src', self._img_ph)
            ts_rand = tf.summary.image('train/rand', tf.clip_by_value(self._stub.rand_img, 0, 1))
            ts_loss = tf.summary.merge([ts_comb_loss, ts_latent_loss, ts_adv_loss, ts_recon_loss])
            ts_imgs = tf.summary.merge([ts_recon, ts_src, ts_rand])

            vs_comb_loss = tf.summary.scalar('val/comb_loss', self._stub.combined_loss)
            vs_latent_loss = tf.summary.scalar('val/latent_loss', self._stub.latent_loss)
            vs_adv_loss = tf.summary.scalar('val/adv_loss', self._stub.adv_loss)
            vs_recon_loss = tf.summary.scalar('val/recon_loss', self._stub.recon_loss)
            vs_recon = tf.summary.image('val/recon', tf.clip_by_value(self._stub.recon_img, 0, 1))
            train_sum_rand = tf.summary.image('val/rand', tf.clip_by_value(self._stub.rand_img, 0, 1))
            vs_src = tf.summary.image('val/src', self._img_ph)
            vs_loss = tf.summary.merge([vs_comb_loss, vs_latent_loss, vs_recon_loss, vs_adv_loss])
            vs_imgs = tf.summary.merge([vs_recon, vs_src, train_sum_rand])

            self._summ = StyleVaeSummary(vs_loss, vs_imgs, ts_loss, ts_imgs)

    def train(self):
        fetches = self._stub.get_train_fetch()

        for e in range(self._config.num_epochs):
            self.EPOCH = e
            self._sess.run(self._iter_init)
            self._run_epoch(fetches, phase='train')
            self.validate()
            self.save()

    def _run_epoch(self, fetches, phase):
        if phase == 'train':
            fetches['summary'] = self._summ.get_loss_sum(phase)
        fetches['global_step'] = self._global_step
        steps = self._num_train if phase == 'train' else self._num_val
        progress = tqdm(range(steps))
        avg_loss = 0
        for step in progress:
            res = self._sess.run(fetches)
            loss_c, loss_r, loss_l = res['comb_loss'], res['recon_loss'], res['latent_loss']
            progress.set_description(f'{phase} epoch {self.EPOCH:^3}/{self._config.num_epochs:3}|{step:^4}/{steps:^4}'
                                     f' | loss: {loss_c:^.4}, recon_loss: {loss_r:^.4}, latent_loss: {loss_l:^.4}')
            avg_loss += loss_c
            self._graph_writer.add_summary(res['summary'], global_step=res['global_step'])

            if step == len(progress) - 1:
                loss_summary_op = self._summ.get_loss_sum(phase)
                img_summary_op = self._summ.get_img_sum(phase)
                img_summary, loss_summary, global_step = self._sess.run([img_summary_op, loss_summary_op, self._global_step])
                self._graph_writer.add_summary(loss_summary, global_step=global_step)
                self._graph_writer.add_summary(img_summary, global_step=global_step)

        avg_loss /= steps
        print(f'\n{phase} epoch {self.EPOCH:3}/{self._config.num_epochs:3} --> avg_loss: {avg_loss:.4}')

    def validate(self):
        fetches = self._stub.get_validate_fetch()
        fetches['summary'] = self._summ.get_loss_sum('validate')
        self._run_epoch(fetches, phase='val')

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

    def get_model(self) -> StyleVae:
        return self._model

    def get_decode_stubs(self):
        return self._code_ph, self._decoded
