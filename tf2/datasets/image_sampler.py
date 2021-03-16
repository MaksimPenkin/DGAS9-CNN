""" 
 @author   Maksim Penkin
"""


import numpy as np
import random
import os


class ImagePathSampler:
    def __init__(self, image_list,
                 basedir = '',
                 shuffle = True):

        self.basedir = basedir
        with open(image_list, 'rt') as f:
            self.image_list = f.read().splitlines()
        self.image_list = list(map(lambda x: [os.path.join(self.basedir, y) for y in x.split(' ')], self.image_list))
        
        self.num_samples = len(self.image_list)
        
        self.shuffle = shuffle
        self._reset()

    def _reset(self):
        self.pos = 0
        if self.shuffle:
            random.shuffle(self.image_list)

    def __iter__(self):
        self._reset()
        return self

    def __len__(self):
        return self.num_samples

    def __next__(self):
        if self.pos >= self.num_samples:
            self._reset()

        one_sample = self.image_list[self.pos]
        self.pos = self.pos + 1
        if len(one_sample) == 1:
            return one_sample[0]
        else:
            return tuple(one_sample)

    def __call__(self):
        return self.__iter__()


