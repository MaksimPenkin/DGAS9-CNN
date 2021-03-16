""" 
 @author   Maksim Penkin
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from models.base_layer import BaseLayer


class Activation(BaseLayer):

    AVAILABLE_NAMES = [
        'relu', 'lrelu', 'elu',
        'prelu', 'none'
    ]

    def __init__(self, activate_fn,
                 name = 'activation',
                 trainable = True):
        super(Activation, self).__init__(name=name, trainable=trainable)

        self.activate_fn = activate_fn
        assert self.activate_fn in self.AVAILABLE_NAMES

        if self.activate_fn == 'relu':
            self.func = tf.keras.layers.ReLU(name='relu')

        elif self.activate_fn == 'lrelu':
            self.func = tf.keras.layers.LeakyReLU(name='lrelu')

        elif self.activate_fn == 'elu':
            self.func = tf.keras.layers.ELU(name='elu')

        elif self.activate_fn == 'prelu':
            init_fn = tf.random_normal_initializer(stddev=2 / (3 * 3))
            self.func = tf.keras.layers.PReLU(alpha_initializer=init_fn,
                                              shared_axes=[1, 2],
                                              name='prelu',
                                              trainable=self.trainable)
        else:
            self.func = tf.keras.layers.Lambda(lambda x: x)

    def call(self, x, **kwargs):
        y = self.func(x)
        return y

class MegviiResidual(BaseLayer):
    def __init__(self, num_filters,
                 activate_fn,
                 name = 'megvii_residual',
                 use_batchnorm = False,
                 regularization = dict(mode='l2', weight=0.0001),
                 trainable = True):
        super(MegviiResidual, self).__init__(name=name, trainable=trainable)
        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm

        # Sub-block 1
        if self.use_batchnorm:
            self.batchnorm1 = BatchNormalization(trainable=self.trainable, name='sub1_batchnorm')
        self.activate_fn1 = Activation(activate_fn=activate_fn,
                                       name='sub1_activation',
                                       trainable=self.trainable)
        self.conv1 = Conv2D(filters=self.num_filters,
                             kernel_size=3, strides=(1, 1),
                             padding='same', activation=None,
                             name='sub1_conv',
                             use_bias=True, kernel_regularizer=get_regularizer(**regularization), trainable=self.trainable)

        # Sub-block 2
        if self.use_batchnorm:
            self.batchnorm2 = BatchNormalization(trainable=self.trainable, name='sub2_batchnorm')
        self.activate_fn2 = Activation(activate_fn=activate_fn,
                                       name='sub2_activation',
                                       trainable=self.trainable)
        self.conv2 = Conv2D(filters=self.num_filters,
                             kernel_size=3, strides=(1, 1),
                             padding='same', activation=None,
                             name='sub2_conv',
                             use_bias=True, kernel_regularizer=get_regularizer(**regularization), trainable=self.trainable)

    def call(self, x, training=True, **kwargs):
        original_x = x
        if self.use_batchnorm:
            x = self.batchnorm1(x, training=training)
        x = self.activate_fn1(x)
        x = self.conv1(x)

        if self.use_batchnorm:
            x = self.batchnorm2(x, training=training)
        x = self.activate_fn2(x)
        x = self.conv2(x)

        y = x + original_x
        return y

class FusionConcat(BaseLayer):
    def __init__(self, num_filters,
                 activate_fn,
                 name = 'fusion_concat',
                 use_batchnorm = False,
                 regularization = dict(mode='l2', weight=0.0001),
                 trainable = True):
        super(FusionConcat, self).__init__(name=name, trainable=trainable)
        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            self.batchnorm = BatchNormalization(trainable=self.trainable, name='batchnorm')
        self.activate_fn = Activation(activate_fn=activate_fn,
                                       name='activation',
                                       trainable=self.trainable)
        self.conv_proj = Conv2D(filters=self.num_filters,
                             kernel_size=3, strides=(1, 1),
                             padding='same', activation=None,
                             name='conv_proj',
                             use_bias=True, kernel_regularizer=get_regularizer(**regularization), trainable=self.trainable)

    def call(self, x, y, training=True, **kwargs):
        z = tf.concat([x, y], axis=3)
        if self.use_batchnorm:
            z = self.batchnorm(z, training=training)
        z = self.activate_fn(z)
        z = self.conv_proj(z)

        return z

def get_regularizer(mode, weight):
    AVAILABLE_REG_MODES = ['l1', 'l2']
    assert mode in AVAILABLE_REG_MODES

    if mode == 'l1':
        return tf.keras.regularizers.L1(l1=weight)
    elif mode == 'l2':
        return tf.keras.regularizers.L2(l2=weight)



