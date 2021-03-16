""" 
 @author   Maksim Penkin
"""

import tensorflow as tf


class Dataset:
    def __init__(self, image_list,
                 db_basedir='',
                 batch_size=8,
                 patch_size=None,
                 num_random_crops=None,
                 num_parallel_calls=4,
                 buffer_size=400,
                 prefetch=1,
                 shuffle=True):

        self.image_list = image_list
        self.db_basedir = db_basedir
        self.batch_size = batch_size 
        self.patch_size = patch_size 
        self.num_random_crops = num_random_crops
        self.num_parallel_calls = num_parallel_calls
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.shuffle = shuffle

        self.dtype = tf.float32

        self._dataset = None
        self.num_samples = None
        self.num_batches = None

    @property
    def dataset(self):
        return self._dataset

    def __iter__(self):
        return iter(self.dataset)


