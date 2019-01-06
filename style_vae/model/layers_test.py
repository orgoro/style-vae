from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# 3rd party:
import tensorflow as tf
import numpy as np
import unittest

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
        noise = tf.ones((batch, size * 2, size * 2, 1))
        style = tf.ones((batch, 4))

        # ACT:
        f_maps = 2
        scaled_up = VaeLayers.cell_up(x, f_maps, noise, style)
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
        x = tf.random_normal((batch, size, size, f_maps), mean=10, stddev=5)

        # ACT:
        normed = VaeLayers.normalize(x)
        self.sess.run(tf.global_variables_initializer())
        result = self.sess.run(normed)

        # ASSERT:
        np.testing.assert_almost_equal(np.mean(result, axis=3), np.zeros((batch, size, size)), 3, 'zero mean')
        np.testing.assert_almost_equal(np.var(result, axis=3), np.ones((batch, size, size)), 3, 'unit var')
        self.assertEqual(result.shape, (batch, size, size, f_maps), 'shape must not change')


if __name__ == '__main__':
    unittest.main()
