""" 
 @author   Maksim Penkin
"""

import numpy as np
import cv2
import tensorflow as tf

from datasets.dataset import Dataset
from datasets.image_sampler import ImagePathSampler
from datasets.transform import read_input_images


class HybridDataset(Dataset):
    def __init__(self, ch, mode='train', **kwargs):
        super(HybridDataset, self).__init__(**kwargs)
        self.ch = ch

        path_generator = ImagePathSampler(self.image_list, basedir=self.db_basedir, shuffle=self.shuffle)

        if mode == 'train':
            self.num_samples = path_generator.num_samples
            self.num_batches = int(self.num_samples * self.num_random_crops / self.batch_size)
            
            self._dataset = (tf.data.Dataset.from_generator(
                                path_generator, 
                                output_signature=(
                                    tf.TensorSpec(shape=(), dtype=tf.string),
                                    tf.TensorSpec(shape=(), dtype=tf.string),
                                    tf.TensorSpec(shape=(), dtype=tf.string)))
                                .map(lambda path_gt, path_sketch, path_kellner: self.load_images(path_gt, path_sketch, path_kellner), num_parallel_calls=self.num_parallel_calls)
                                .map(lambda gt, sketch, kellner: self.crop_patches(gt, sketch, kellner), num_parallel_calls=self.num_parallel_calls)
                                .unbatch()
                                .map(lambda x: tf.cast(x, self.dtype))
                                .shuffle(buffer_size=self.buffer_size)
                                .batch(batch_size=self.batch_size)
                                .prefetch(self.prefetch)
                                .map(lambda HL: (HL[:, :, :, :self.ch], HL[:, :, :, self.ch:(self.ch+self.ch)],  HL[:, :, :, (self.ch+self.ch):]), num_parallel_calls=self.num_parallel_calls)
                            )
        elif (mode == 'val') or (mode == 'test'):
            self.num_samples = path_generator.num_samples
            self.num_batches = int(self.num_samples  / self.batch_size)
            
            self._dataset = (tf.data.Dataset.from_generator(
                                path_generator, 
                                output_signature=(
                                    tf.TensorSpec(shape=(), dtype=tf.string),
                                    tf.TensorSpec(shape=(), dtype=tf.string),
                                    tf.TensorSpec(shape=(), dtype=tf.string)))
                                .map(lambda path_gt, path_sketch, path_kellner: self.load_images(path_gt, path_sketch, path_kellner, do_aug=False), num_parallel_calls=self.num_parallel_calls)
                                .map(lambda x, y, z: (tf.cast(x, self.dtype), tf.cast(y, self.dtype), tf.cast(z, self.dtype)))
                                .batch(batch_size=self.batch_size)
                                .prefetch(self.prefetch)
                            )

    def crop_patches(self, gt, sketch, kellner):
        patches = []
        cat = tf.concat([gt, sketch, kellner], -1)

        for i in range(self.num_random_crops):
            patch = tf.image.random_crop(cat, [self.patch_size, self.patch_size, self.ch + self.ch + self.ch])
            patches.append(patch)
        patches = tf.stack(patches)
        assert patches.get_shape().dims == [self.num_random_crops, self.patch_size, self.patch_size, self.ch + self.ch + self.ch]

        return patches

    def load_images(self, path_gt, path_sketch, path_kellner, do_aug=True):
        rot = np.random.randint(low=0, high=4) % 4
        doflip = bool(np.random.randint(low=0, high=2))
        flip_axis = np.random.randint(low=0, high=2) % 2
        
        img_gt, img_sketch, img_kellner = tf.numpy_function(read_input_images, [path_gt, path_sketch, path_kellner, do_aug, rot, doflip, flip_axis], (tf.float32, tf.float32, tf.float32))

        return img_gt, img_sketch, img_kellner







