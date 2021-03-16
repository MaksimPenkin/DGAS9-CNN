""" 
 @author   Maksim Penkin
"""

import tensorflow as tf


class L1(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, name="l1"):
        super(L1, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.weight = weight

    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis=(1, 2, 3))
        return loss

class L2(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, name="l2"):
        super(L2, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.weight = weight

    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2, 3))
        return loss

class PSNR(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, name="psnr"):
        super(PSNR, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.weight = weight

    def call(self, y_true, y_pred):
        loss = tf.image.psnr(
                        tf.clip_by_value(y_true, 0.0, 1.0), 
                        tf.clip_by_value(y_pred, 0.0, 1.0),
                        max_val=1.0)
        return loss

class SSIM(tf.keras.losses.Loss):
    def __init__(self, weight=1.0, name="ssim"):
        super(SSIM, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.weight = weight

    def call(self, y_true, y_pred):
        ssim_diff = tf.image.ssim(y_true, y_pred, max_val=1.0)
        loss = tf.clip_by_value(0.5 * (1.0 - ssim_diff), 0.0, 1.0)
        return loss

def calc_generator_loss(loss_regression_objects, gt, generated_output, loss_regularizarion=None):
    losses = {}
    total_loss = 0.0

    for loss_obj in loss_regression_objects:
        loss = tf.reduce_mean(loss_obj(gt, generated_output))        
        losses[loss_obj.name] = loss
        total_loss += loss * loss_obj.weight
    if loss_regularizarion is not None:
        losses['regularization'] = tf.reduce_mean(loss_regularizarion)
        total_loss += loss_regularizarion

    losses['G_total_loss'] = total_loss
    return total_loss, losses


