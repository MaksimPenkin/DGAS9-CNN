import numpy as np
import cv2
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

def read_input_images(path_gt, path_sketch, path_kellner, do_aug=True, rot=0, doflip=False, flip_axis=0):
    img_gt = np.load(path_gt).astype(np.float32)
    img_sketch = np.load(path_sketch)
    img_kellner = np.load(path_kellner)

    params = {}
    params['sub'] = np.amin(img_gt)
    params['scale'] = np.amax(img_gt) - np.amin(img_gt)
    
    img_gt = data_normaliser(img_gt, mode='byvalue', value=params)
    img_sketch = data_normaliser(img_sketch, mode='byvalue', value=params)
    img_sketch = np.abs(img_sketch).astype(np.float32)
    img_kellner = data_normaliser(img_kellner, mode='maxmin')

    img_kellner = cv2.resize(np.abs(img_kellner).astype(np.float32), (256,256), cv2.INTER_NEAREST)[..., np.newaxis]

    if do_aug:
        flip_params = {'doflip':doflip, 'flip_axis':flip_axis}
        img_gt = augmentation(img_gt, rot, flip_params)
        img_sketch = augmentation(img_sketch, rot, flip_params)
        img_kellner = augmentation(img_kellner, rot, flip_params)

    return img_gt, img_sketch, img_kellner




