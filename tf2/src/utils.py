""" 
 @author   Maksim Penkin
"""

import argparse
import os, shutil, json
import time
from datetime import datetime
import numpy as np
import cv2
from collections.abc import MutableMapping
import tensorflow as tf

from datasets.transform import touint8, touint16


def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive int value' % value)
    return ivalue

def check_positive_float(value):
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('%s is an invalid positive float value' % value)
    return ivalue

def parse_args():
    parser = argparse.ArgumentParser(description='Gibbs-ringing suppression arguments', usage='%(prog)s [-h]')
    
    parser.add_argument('--stage', type=str, default='apply', choices=['train', 'apply', 'to_proto', 'to_saved_model'],
                        help='stage: train, apply, to_proto, to_saved_model', metavar='')
    
    parser.add_argument('--checkpoints', type=str, default=r'E:\Gibbs_ringing\code\tf2\checkpoints',
                        help='path to train directories; here the train directory of the current model will be created: checkpoints and other stuff', metavar='')
    parser.add_argument('--postfix', type=str, default='test_1', 
                        help='postfix to create checkpoint directory.', metavar='')
    parser.add_argument('--train_force', default=False, action='store_true',
                        help='if set, overwrite the train directory of the current model')
    
    parser.add_argument('--db_regression_name', type=str, default='IXI_0_1', 
                        help='name of the train/val regression dataset', metavar='')
    parser.add_argument('--db_regression_postfix', type=str, default='hybrid', 
                        help='postfix to create path to regression train/val dataset list', metavar='')
    parser.add_argument('--db_regression_basedir', type=str, default='', 
                        help='basedir of the regression dataset. If provided, then train/val lists store relative pathes to make regression dataset mobile', metavar='')

    parser.add_argument('--train_lists', type=str, default=r'E:\Gibbs_ringing\data\datalists\tf2', 
                        help='directory with train/val dataset lists', metavar='')
    parser.add_argument('--validate_from_txt', default=False, action='store_true',
                        help='if set, turn on validation from *.txt file, otherwise, no validation is performed by default')
    
    parser.add_argument('--batch_size', type=check_positive_int, default=8,
                        help='batch size, feeding into the model', metavar='')
    parser.add_argument('--patch_size', type=check_positive_int, default=48,
                        help='patch size, feeding into the model', metavar='')
    parser.add_argument('--ch', type=check_positive_int, default=1,
                        help='number of channels in images', metavar='')
    
    parser.add_argument('--num_random_crops', type=check_positive_int, default=8,
                        help='number of random crops from each image while passing through CNN', metavar='')
    parser.add_argument('--num_parallel_calls', type=check_positive_int, default=4,
                        help='number of parallel calls for loading images while passing through CNN', metavar='')
    parser.add_argument('--buffer_size', type=check_positive_int, default=400,
                        help='buffer size for shuffling image-patches', metavar='')
    parser.add_argument('--prefetch', type=check_positive_int, default=1,
                        help='keeping number of ready-to-go batches', metavar='')
    
    parser.add_argument('--epoch_number', type=check_positive_int, default=600,
                        help='number of training epochs', metavar='')
    parser.add_argument('--learning_rate', type=check_positive_float, default=1e-4,
                        help='learning rate value', metavar='')

    parser.add_argument('--apply_list', type=str, default='', 
                        help='full path to the apply *.txt list', metavar='')
    parser.add_argument('--apply_savedir', type=str, default='', 
                        help='directory for saving resulting images on apply stage', metavar='')
    parser.add_argument('--apply_force', default=False, action='store_true',
                        help='if set, overwrite existed apply save directory')
    parser.add_argument('--bitdepth_out', type=check_positive_int, default=8, choices=[8, 16],
                        help='bitdepth of output images on apply stage', metavar='')
    
    parser.add_argument('--restore_epoch', type=check_positive_int,
                        help='epoch number to be restored', metavar='')
    parser.add_argument('--restore_step', type=check_positive_int,
                        help='step number to be restored', metavar='')
    parser.add_argument('--restore_checkpoint_template', type=str, default='G_e{:04d}_s{:08d}',
                        help='checkpoint template to be restored: e - epoch; s - step', metavar='')

    args = parser.parse_args()
    return args

def delete_file_folder(f):
    try:
        if os.path.isfile(f) or os.path.islink(f):
            os.unlink(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (f, e))

def delete_contents_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        delete_file_folder(file_path)

def create_folder(folder, force=False, raise_except_if_exists=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if force:
            delete_contents_folder(folder)
        else:
            if raise_except_if_exists:
                raise Exception('utils.py: def create_folder(...): error: directory {} exists. In order to overwrite it set force=True'.format(folder))

def save_batch(batch, save_dir, dtype=np.uint8, create_dir=True, force=False, raise_except_if_exists=True, names_batch=None):
    if create_dir:
        create_folder(save_dir, force=force, raise_except_if_exists=raise_except_if_exists)
    for i,img in enumerate(batch):
        if dtype == np.uint8:
            img = touint8(img)
        elif dtype == np.uint16:
            img = touint16(img)

        if names_batch is not None:
            cv2.imwrite(os.path.join(save_dir, '{}'.format(names_batch[i])), img)
        else:
            cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i)), img)

def get_dummy_tensor(batch_size=8, height=48, width=48, channels=1, dtype=tf.float32):
    return tf.random.normal(shape=(batch_size, height, width, channels), dtype=dtype)

def get_training_str(G_losses, epoch, epoch_max, step, step_max):
    res_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f': Epoch {epoch}/{epoch_max}, Step {step}/{step_max};'
    for l_name, l_value in G_losses.items():
        res_str += f' {l_name}: {l_value:.5f};'
    return res_str

def save_logs(summary_writer, G_losses, step):
    with summary_writer.as_default():
        for l_name, l_value in G_losses.items():
            tf.summary.scalar(name=l_name, data=l_value, step=step)

def dump2json(obj, fp):
    with open(fp, 'w') as f:
        json.dump(obj, f, indent=4)

class AccumulativeDict(MutableMapping):
    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)
    
    def __len__(self):
        return len(self.store)

    def __str__(self):
        res_str = ''
        for k,v in self.items():
            res_str += f'{k}: {v:.5f};\n'
        return res_str

    def __add__(self, value):
        result = dict() # create resulting object
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            # if first term is empty
            if not self.keys():
                return AccumulativeDict(value) # return copy of second term as a resulting object
            # else, get overall keys
            k_overall = self.keys() | value.keys()
            # foreach key in the overall collection
            for k in k_overall:
                a = self[k] if k in self.keys() else 0 # take first arg-of-sum
                b = value[k] if k in value.keys() else 0 # take second arg-of-sum
                result[k] = a + b # accumulate them in the resulting object
        else:
            for k,v in self.items():
                result[k] = self[k] + value
        return AccumulativeDict(result)

    def __neg__(self):
        result = dict()
        for k,v in self.items():
            result[k] = -v
        return result
    
    def __sub__(self, value):
        result = dict() # create resulting object
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            # if first term is empty
            if not self.keys():
                return -AccumulativeDict(value) # return copy -second term as a resulting object
            # else, get intersection of keys
            k_intersect = self.keys() & value.keys()
            # foreach term in intersection
            for k in k_intersect:
                result[k] = self[k] - value[k] # save the difference as for resulting object
        else:
            for k,v in self.items():
                result[k] = self[k] - value
        return AccumulativeDict(result)

    def __mul__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] * value[k]
        else:
            for k,v in self.items():
                result[k] = self[k] * value
        return AccumulativeDict(result)

    def __floordiv__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] // value[k]
        else:
            for k,v in self.items():
                result[k] = self[k] // value
        return AccumulativeDict(result)

    def __truediv__(self, value):
        result = dict()
        if isinstance(value, dict) or isinstance(value, AccumulativeDict):
            if (not self.keys()) or (not value.keys()):
                return AccumulativeDict({})
            k_intersect = self.keys() & value.keys()
            for k in k_intersect:
                result[k] = self[k] / value[k]
        else:
            for k,v in self.items():
                result[k] = self[k] / value
        return AccumulativeDict(result)


