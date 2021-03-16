""" 
 @author   Maksim Penkin
"""

import tensorflow as tf
from models.embedding import ConvEmbedding, ConvEmbedding_multistep
from models.encoder import ResidualEncoder
from models.decoder import ResidualFusionDecoder
from models.restoration import ResidualRestoration_multistep
from models.base_model import BaseModel


class ResidualFusionGenerator(BaseModel):
    def __init__(self,
                 num_filters = 64,
                 num_blocks = 3,
                 ch = 1,
                 name = 'residual_fusion_generator',
                 trainable = True):
        super(ResidualFusionGenerator, self).__init__(name=name, trainable=trainable)

        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.ch = ch

        # Embedding part
        self.pre_conv_sketch = ConvEmbedding(num_filters=self.num_filters,
                                        activate_fn='none',
                                        kernel_size = 3,
                                        name = 'pre_conv_sketch',
                                        use_batchnorm = True,
                                        trainable = self.trainable)
        self.pre_conv_kellner = ConvEmbedding_multistep(num_filters_lst=[1, 1, self.num_filters],
                                        activate_fn_lst=['relu', 'relu', 'none'],
                                        kernel_size_lst = [3, 3, 3],
                                        name = 'pre_conv_kellner',
                                        use_batchnorm_lst = [True, True, True],
                                        trainable = self.trainable)

        # Encode acts from sketch and from kellner
        self.encoder_sketch = ResidualEncoder(num_filters=self.num_filters,
                                        activate_fn='relu',
                                        num_blocks = self.num_blocks,
                                        name = 'residual_encoder_sketch',
                                        use_batchnorm = True,
                                        trainable = self.trainable)
        self.encoder_kellner = ResidualEncoder(num_filters=self.num_filters,
                                        activate_fn='relu',
                                        num_blocks = self.num_blocks,
                                        name = 'residual_encoder_kellner',
                                        use_batchnorm = True,
                                        trainable = self.trainable)

        # Decode features, by fusing encoded acts of sketch and kellner
        self.decoder = ResidualFusionDecoder(num_filters=self.num_filters,
                                        activate_fn='relu',
                                        num_blocks = self.num_blocks,
                                        name = 'residual_fusion_decoder',
                                        use_batchnorm = True,
                                        trainable = self.trainable)

        # Restore final image
        self.post_conv = ResidualRestoration_multistep(num_filters_lst=[16, self.ch],
                                        activate_fn_lst=['relu', 'none'],
                                        kernel_size_lst = [3, 3],
                                        name = 'post_conv',
                                        use_batchnorm_lst = [True, False],
                                        trainable = self.trainable)

    def call(self, inputs, training=True, **kwargs):
        sketch = inputs[:,:,:,:self.ch]
        kellner = inputs[:,:,:,self.ch:]

        # Embedding
        emb_sketch = self.pre_conv_sketch(sketch, training=training)
        emb_kellner = self.pre_conv_kellner(kellner, training=training)

        # Encoder 
        encoder_act_sketch = self.encoder_sketch(emb_sketch, training=training)
        encoder_act_kellner = self.encoder_kellner(emb_kellner, training=training)

        # Decoder
        decoder_act = self.decoder(encoder_act_sketch, encoder_act_kellner, training=training)

        # Restoration
        y = self.post_conv(decoder_act, sketch, training=training)
        return y


