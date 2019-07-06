

# 3rd party:
import tensorflow as tf
import unittest

# tested file:
from style_vae.model.style_vae import StyleVae, Config


class StyleVaeTester(unittest.TestCase):
    sess = None
    style_vae = None

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()
        cls.sess = tf.Session()
        cls.style_vae = StyleVae(Config(code_size=10, img_dim=256, batch_size=3, num_channels=3))

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_when_encode_decode_then_same_size(self):
        # ARRANGE:
        size = 256
        batch = 3
        img = tf.ones((batch, size, size, 3))

        # ACT:
        code_mean, code_log_std = self.style_vae.encode(img)
        code, _ = self.style_vae.map(code_mean, code_log_std)
        recon_img = self.style_vae.decode(code)

        # ASSERT:
        code_shape = (self.style_vae.config.batch_size, self.style_vae.config.code_size)
        self.assertEqual(code.shape, code_shape, 'shape must be BxK')
        self.assertEqual(recon_img.shape, img.shape, 'shape must be same')


if __name__ == '__main__':
    unittest.main()
