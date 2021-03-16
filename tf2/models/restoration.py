""" 
 @author   Maksim Penkin
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from models.layers import Activation, get_regularizer
from models.base_model import BaseModel


class ResidualRestoration(BaseModel):
    def __init__(self, num_filters,
                 activate_fn,
                 kernel_size = 3,
                 name = 'residual_restore',
                 use_batchnorm = False,
                 regularization = dict(mode='l2', weight=0.0001),
                 trainable = True):
        super(ResidualRestoration, self).__init__(name=name, trainable=trainable)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm

        if self.use_batchnorm:
            self.batchnorm = BatchNormalization(trainable=self.trainable, name='batchnorm')
        self.activate_fn = Activation(activate_fn=activate_fn,
                                       name='activation',
                                       trainable=self.trainable)
        self.conv = Conv2D(filters=self.num_filters,
                            kernel_size=self.kernel_size, strides=(1, 1),
                            padding='same', activation=None,
                            name='conv',
                            use_bias=True, kernel_regularizer=get_regularizer(**regularization), trainable=self.trainable)

    def call(self, x, sub, training=True, **kwargs):
        if self.use_batchnorm:
            x = self.batchnorm(x, training=training)
        x = self.activate_fn(x)
        x = self.conv(x)
        y = x + sub
        return y

class ResidualRestoration_multistep(BaseModel):
    def __init__(self, num_filters_lst,
                 activate_fn_lst,
                 kernel_size_lst,
                 use_batchnorm_lst,
                 name = 'residual_restore_multistep',
                 regularization = dict(mode='l2', weight=0.0001),
                 trainable = True):
        super(ResidualRestoration_multistep, self).__init__(name=name, trainable=trainable)

        self.num_filters_lst = num_filters_lst
        self.activate_fn_lst = activate_fn_lst
        self.kernel_size_lst = kernel_size_lst
        self.use_batchnorm_lst = use_batchnorm_lst
        assert len(self.num_filters_lst) == len(self.activate_fn_lst) == len(self.kernel_size_lst) == len(self.use_batchnorm_lst)
        self.num_restore_steps = len(self.num_filters_lst)

        self.bn_layers = {}
        self.activate_layers = {}
        self.conv_layers = {}

        for i in range(self.num_restore_steps):
            if self.use_batchnorm_lst[i]:
                self.bn_layers['bn_{}'.format(i+1)] = BatchNormalization(trainable=self.trainable, name='bn_{}'.format(i+1))
            self.activate_layers['activate_{}'.format(i+1)] = Activation(activate_fn=self.activate_fn_lst[i], name='activation_{}'.format(i+1), trainable=self.trainable)
            self.conv_layers['conv_{}'.format(i+1)] = Conv2D(filters=self.num_filters_lst[i],
                                                            kernel_size=self.kernel_size_lst[i], strides=(1, 1),
                                                            padding='same', activation=None,
                                                            name='conv_{}'.format(i+1),
                                                            use_bias=True, kernel_regularizer=get_regularizer(**regularization),
                                                            trainable=self.trainable)

    def call(self, x, sub, training=True, **kwargs):
        for i in range(self.num_restore_steps):
            if self.use_batchnorm_lst[i]:
                x = self.bn_layers['bn_{}'.format(i+1)](x, training=training)
            x = self.activate_layers['activate_{}'.format(i+1)](x)
            x = self.conv_layers['conv_{}'.format(i+1)](x)
        y = x + sub
        return y


