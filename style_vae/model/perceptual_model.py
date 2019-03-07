import tensorflow as tf
from tensorflow.python.keras import Model, layers
from tensorflow.python.keras import applications as apps
from typing import List


class PerceptualModel(object):
    Types = ['vgg16', 'vgg19']
    _type_2_layer = {'vgg16': ['block1_conv1', 'block2_conv1', 'block3_conv1'],
                     'vgg19': ['block1_conv1', 'block2_conv1', 'block3_conv1']}

    def __init__(self, model_type: str = 'vgg16', layer_names: List[str] = None, img_shape: tuple = (256, 256, 3)):
        """
        :param model_type: model type for underlying model see PerceptualModel.Types default is vgg16
        :param layer_names: list of layer names to create the perceptual feature or None for default
        :param img_shape: WHC shape for input image
        """
        self._model_type = model_type
        self._model = None
        self._feature_model = None
        if layer_names is None:
            layer_names = self._type_2_layer[self._model_type]
        self._layers = layer_names
        self._img_shape = img_shape
        self._build_graph()

    def __call__(self, img: tf.Tensor) -> tf.Tensor:
        return self.get_feature(img)

    def get_feature(self, img: tf.Tensor) -> tf.Tensor:
        """ get perceptual feature from imge
        :param img: image tensor NWHC float32 tensor
        :return: feature (N,feature_len) tensor
        """
        assert len(img.shape) == 4, 'input should be NWHC'
        assert img.shape[3] == self._img_shape[2], 'channels number is incompatible'
        x = img
        with tf.variable_scope('perceptual_model_' + self._model_type):
            # resize and extract feature
            if img.shape[1] != self._img_shape[0] or img.shape[2] != self._img_shape[1]:
                x = tf.image.resize_bicubic(x, self._img_shape[0:2])
            x = self._feature_model(x)

            # flatten
            if isinstance(x, list):
                x = [layers.Flatten()(feature) for feature in x]
                x = tf.concat(x, axis=1)
            else:
                x = layers.Flatten()(x)

        return x

    def _build_graph(self):
        assert self._model_type in self.Types, f'model type unrecognised: {self._model_type} known types: {self.Types}'
        with tf.variable_scope('perceptual_model_' + self._model_type):
            if self._model_type == 'vgg16':
                self._model = apps.VGG16(include_top=False, weights='imagenet', input_shape=self._img_shape)
            elif self._model_type == 'vgg19':
                self._model = apps.VGG19(include_top=False, weights='imagenet', input_shape=self._img_shape)
            outputs = [self._model.get_layer(l).output for l in self._layers]
            self._feature_model = Model(self._model.input, outputs)
            self._freeze_all_vars_in_scope()

    @staticmethod
    def _freeze_all_vars_in_scope():
        """ freezes all variable in current scope by removing them from the tf global collection of trainable vars """
        scope = tf.get_variable_scope()
        vars_to_freeze = tf.trainable_variables(scope.name)
        collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for v in vars_to_freeze:
            collection.remove(v)
