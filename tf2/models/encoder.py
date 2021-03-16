""" 
 @author   Maksim Penkin
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from models.layers import MegviiResidual
from models.base_model import BaseModel


class ResidualEncoder(BaseModel):
    def __init__(self, num_filters,
                 activate_fn,
                 num_blocks = 3,
                 name = 'residual_encoder',
                 use_batchnorm = False,
                 trainable = True):
        super(ResidualEncoder, self).__init__(name=name, trainable=trainable)

        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm
        self.num_blocks = num_blocks

        self.act_layers = {}

        for i in range(self.num_blocks):
            self.act_layers['megvii_residual_{}'.format(i+1)] = MegviiResidual(self.num_filters,
                                                                                activate_fn,
                                                                                name = 'megvii_residual_{}'.format(i+1),
                                                                                use_batchnorm = self.use_batchnorm,
                                                                                trainable = self.trainable)

    def call(self, x, training=True, **kwargs):
        acts = []
        acts.append(x)
        for i in range(self.num_blocks):
            act = self.act_layers['megvii_residual_{}'.format(i+1)](x, training=training)
            acts.append(act)

        return acts
        

