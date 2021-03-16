import numpy as np
import tensorflow as tf


def augmentation(img, rot, flip_params):
    if rot!=0:
        img = np.rot90(img, k=rot)
    if flip_params['doflip']:
        img = np.flip(img, axis=flip_params['flip_axis'])

    return img

def data_normaliser(data, mode='maxmin', value=None):
    if mode=='maxmin':
        return (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    elif mode=='standart_scaler':
        return (data - np.mean(data)) / np.std(data)
    elif mode=='byvalue':
        return (data - value['sub']) / value['scale']
    else:
        raise Exception('transform.py: def data_normaliser(...): error: mode can be only [maxmin, standart_scaler, byvalue], found {}'.format(mode))

def touint8(img):
    coef = 2**8 - 1
    return (np.clip(img, 0., 1.)*coef).astype(np.uint8)

def touint16(img):
    coef = 2**16 - 1
    return (np.clip(img, 0., 1.)*coef).astype(np.uint16)





