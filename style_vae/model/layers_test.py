from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import tensorflow as tf
import numpy as np
import unittest
from scipy import signal

# tested file:
from style_vae.model.layers import VaeLayers


class LayersTester(unittest.TestCase):
    sess = None

    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()
        cls.sess = tf.Session()

    @classmethod
    def tearDownClass(cls):
        cls.sess.close()

    def test_when_cell_up_then_scale_x2(self):
        # ARRANGE:
        size = 8
        batch = 10
        x = tf.ones((batch, size, size, 3))
        style = tf.ones((batch, 4))

        # ACT:
        f_maps = 2
        scaled_up = VaeLayers.cell_up(x, f_maps, style)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(scaled_up)

        # ASSERT:
        self.assertEqual(result.shape, (batch, size * 2, size * 2, f_maps), 'shape must be b x 2Size x 2Size')

    def test_when_cell_down_then_scale_half(self):
        # ARRANGE:
        size = 8
        batch = 10
        x = tf.ones((batch, size, size, 3))

        # ACT:
        f_maps = 2
        scaled_down = VaeLayers.cell_down(x, f_maps)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(scaled_down)

        # ASSERT:
        self.assertEqual(result.shape, (batch, size // 2, size // 2, f_maps), 'shape must be b x 1/2Size x 1/2Size')

    def test_when_normalize_then_unit(self):
        # ARRANGE:
        size = 8
        batch = 10
        f_maps = 15
        x = tf.random_normal((batch, size, size, f_maps), mean=10, stddev=5, dtype=tf.float32)

        # ACT:
        normed = VaeLayers.normalize(x)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(normed)

        # ASSERT:
        np.testing.assert_almost_equal(np.mean(result, axis=(1, 2)), np.zeros((batch, f_maps)), 3, 'zero mean')
        np.testing.assert_almost_equal(np.var(result, axis=(1, 2)), np.ones((batch, f_maps)), 3, 'unit var')
        self.assertEqual(result.shape, (batch, size, size, f_maps), 'shape must not change')

    def test_when_blur_then_like_scipy(self):
        # ARRANGE:
        size = 8
        batch = 10
        f_maps = 15
        source = np.random.normal(size=(batch, size, size, f_maps))
        x = tf.constant(source, dtype=tf.float32)

        # ACT:
        blurred = VaeLayers.blur_2d(x)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(blurred)

        # ASSERT:
        kernel = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        kernel /= np.sum(kernel)

        for f in range(f_maps):
            for b in range(batch):
                blurred_scipy = signal.convolve2d(source[b, :, :, f], kernel, mode='same')
                np.testing.assert_almost_equal(result[b, :, :, f], blurred_scipy, decimal=3)


if __name__ == '__main__':
    unittest.main()
