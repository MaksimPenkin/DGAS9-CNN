""" 
 @author   Maksim Penkin
"""

import tensorflow as tf


class BaseLayer(tf.keras.layers.Layer):
    def __init__(self,
                 name = 'base_layer',
                 trainable = True):
        super(BaseLayer, self).__init__(name=name, trainable=trainable)

    def __iter__(self):
        return iter(self.submodules)

    def __str__(self):
        str_print = 'Layer name: {}; Trainable {}\n'.format(self.name, self.trainable)
        for m in self:
            str_print += '  Sublayer: {}; Trainable {}\n'.format(m.name, m.trainable)
            for w in m.weights:
                str_print += '    Variable name: {}; Variable shape: {}; Trainable: {}\n'.format(w.name, w.get_shape(), w.trainable)
        return str_print


