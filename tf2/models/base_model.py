""" 
 @author   Maksim Penkin
"""

import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self,
                 name = 'base_model',
                 trainable = True):
        super(BaseModel, self).__init__(name=name, trainable=trainable)

    def __iter__(self):
        return iter(self.layers)

    def __str__(self):
        str_print = 'Model name: {}; Trainable {}\n'.format(self.name, self.trainable)
        for m in self:
            str_print += '  Submodel: {}; Trainable {}\n'.format(m.name, m.trainable)
            for w in m.weights:
                str_print += '    Variable name: {}; Variable shape: {}; Trainable: {}\n'.format(w.name, w.get_shape(), w.trainable)
        return str_print

    def call(self, inputs, **kwargs):
        raise NotImplementedError

    def get_keras_model(self, x):
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


