""" 
 @author   Maksim Penkin
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from models.layers import MegviiResidual, FusionConcat
from models.base_model import BaseModel


class ResidualFusionDecoder(BaseModel):
    def __init__(self, num_filters,
                 activate_fn,
                 num_blocks = 3,
                 name = 'residual_fusion_decoder',
                 use_batchnorm = False,
                 trainable = True):
        super(ResidualFusionDecoder, self).__init__(name=name, trainable=trainable)

        self.num_filters = num_filters
        self.use_batchnorm = use_batchnorm
        self.num_blocks = num_blocks

        self.fusion_layers = {}
        self.act_layers = {}

        for i in range(self.num_blocks, 0, -1):
            self.fusion_layers['fuse_cnv_{}'.format(i)] = FusionConcat(num_filters = self.num_filters,
                                                                        activate_fn = 'relu',
                                                                        name = 'fusion_cnv_{}'.format(i),
                                                                        use_batchnorm = True,
                                                                        trainable = self.trainable)

            self.act_layers['dec_cnv_{}'.format(i)] = MegviiResidual(self.num_filters,
                                                                activate_fn,
                                                                name = 'dec_cnv_{}'.format(i),
                                                                use_batchnorm = self.use_batchnorm,
                                                                trainable = self.trainable)

    def call(self, acts1, acts2, training=True, **kwargs):
        a1 = acts1[-1]
        for i in range(self.num_blocks, 0, -1):
            a2 = acts2[i]
            left = acts1[i-1]

            y = self.fusion_layers['fuse_cnv_{}'.format(i)](a1, a2, training=training)
            y = self.act_layers['dec_cnv_{}'.format(i)](y, training=training)
            a1 = y + left
        return y


