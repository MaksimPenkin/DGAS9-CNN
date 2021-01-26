""" 
 @author   Maxim Penkin

"""

import numpy as np
import csv
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0

        self.loss = 0.
        self.rmse = 0.
        self.mae = 0.
        self.psnr = 0.

    def update(self, result, step=1):
        self.count += step

        self.loss += np.array(result['loss']).sum()
        self.rmse += np.array(result['rmse']).sum()
        self.mae += np.array(result['mae']).sum()
        self.psnr += np.array(result['psnr']).sum()

    def average(self):

        return {'loss' : self.loss / self.count, 'rmse' : self.rmse / self.count, 'mae' : self.mae / self.count, 'psnr': self.psnr / self.count}

def data_normaliser(data, mode='maxmin', value=None):
    if mode=='maxmin':
        return (data - np.amin(data)) / (np.amax(data) - np.amin(data))
    elif mode=='standart_scaler':
        return (data - np.mean(data)) / np.std(data)
    elif mode=='byvalue':
        return (data - value['sub']) / value['scale']
    else:
        print('Runtime error: def data_normaliser(mode={}); mode can be only [maxmin, standart_scaler, byvalue]'.format(mode))
        exit(1)

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

def WriteToTXTFile(filename, mas, mode='a'):
    with open(filename, mode) as file:
        file.write(mas)

def WriteToCSVFile(filename, mas, mode = 'a', delim = ','):
    with open(filename, mode, newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter = delim)
        for line in mas:
            writer.writerow(line)

def float2binary(img, filename):
    newFileByteArray = bytearray(img)
    newFile = open(filename, 'wb')
    newFile.write(newFileByteArray)
    newFile.close()

def binary2float(filename, img_shape):
    data = np.fromfile(filename, dtype=np.float32)
    return np.reshape(data, newshape=img_shape)

def visualize_wights(sess, weights):
    for v in weights:
        value = sess.run(v)
        print(value.shape)
        line_num = value.shape[3]
        col_num = value.shape[2]
        fig, axes = plt.subplots(line_num, col_num)
        # Some setups for figure view.
        fig.subplots_adjust(hspace=0, wspace=0)
        for i in range(line_num):
            for j in range(col_num):
                axes[i,j].xaxis.set_major_locator(plt.NullLocator())
                axes[i,j].yaxis.set_major_locator(plt.NullLocator()) 
        # Normalizing weights as a preparation for visualizion.
        f_max, f_min = np.amax(value), np.amin(value)
        value = ((value - f_min) / (f_max - f_min)).astype(np.float32)
        for i in range(line_num):
            for j in range(col_num):
                axes[i,j].imshow(value[:,:,j,i], cmap='gray')
        plt.show()

