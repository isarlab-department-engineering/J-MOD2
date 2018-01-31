import tensorflow as tf
import keras.backend as K
import numpy as np
import cv2


def rmse_metric(y_true, y_pred):

    y_true = y_true
    y_pred = y_pred
    rmse = K.sqrt(K.mean(K.square((y_true - y_pred))))
    return rmse

def logrmse_metric(y_true, y_pred):

    first_log = K.log(y_pred + 1.)
    second_log = K.log(y_true + 1.)
    return K.sqrt(K.mean(K.mean(K.square(first_log - second_log), axis=0)))
def sc_inv_logrmse_metric(y_true, y_pred):
    first_log = K.log(y_pred + 1.)
    second_log = K.log(y_true + 1.)
    sc_inv_term = K.square(K.mean(K.mean((first_log - second_log), axis=-1)))
    log_term = K.sqrt(K.mean(K.mean(K.square(first_log - second_log), axis=0)))
    return log_term - sc_inv_term
def sc_inv_logrmse_error(y_true, y_pred):
    first_log = np.log(y_pred + 1.)
    second_log = np.log(y_true + 1.)
    sc_inv_term = np.square(np.mean(np.mean((first_log - second_log), axis=-1)))
    log_term = np.mean(np.mean(np.square(first_log - second_log), axis=0))
    return log_term - sc_inv_term

def rmse_error(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    diff = y_true - y_pred
    square = np.square(diff)
    mean = np.mean(np.mean(square, axis=0))
    rmse_error = np.sqrt(mean + 0.00001)
    return rmse_error


def logrmse_error(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    first_log = np.log(y_pred + 1.)
    second_log = np.log(y_true + 1.)
    return np.sqrt(np.mean(np.mean(np.square(first_log - second_log), axis=0)))

def normals_metric(y_true, y_pred):

    y_true = K.variable(y_true)
    y_pred = K.variable(y_pred)

    y_true = K.expand_dims(y_true,0)


    filter_y = K.variable(np.array([[ 0., -0.5 , 0.],
                               [0., 0., 0.],
                               [0., 0.5, 0.]]).reshape(3, 3, 1, 1))


    filter_x = K.variable(np.array([ [0, 0., 0.],
                               [0.5, 0., -0.5],
                               [0., 0., 0.]]).reshape(3, 3, 1, 1))

    dzdx = K.conv2d(K.exp(y_true), filter_x, padding='same')
    dzdy = K.conv2d(K.exp(y_true), filter_y, padding='same')

    dzdx_ = dzdx * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdx))
    dzdy_ = dzdy * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdy))

    mag_norm = K.pow(dzdx,2) + K.pow(dzdy,2) + 1.0#K.constant(1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(1.0, shape=K.int_shape(dzdx))

    mag_norm = K.sqrt(mag_norm)
    N3 = 1.0 / mag_norm #K.constant(1.0, shape=K.int_shape(dzdx)) / mag_norm
    N1 = dzdx_ / mag_norm
    N2 = dzdy_ / mag_norm

    normals = K.concatenate(tensors=[N1,N2,N3],axis=-1)

    dzdx_pred = K.conv2d(K.exp(y_pred), filter_x, padding='same')
    dzdy_pred = K.conv2d(K.exp(y_pred), filter_y, padding='same')

    mag_norm_pred = K.pow(dzdx_pred,2) + K.pow(dzdy_pred,2) + 1.0
    mag_norm_pred = K.sqrt(mag_norm_pred)

    grad_x = K.concatenate(tensors=[1.0/ mag_norm_pred,
                                    0.0/ mag_norm_pred, dzdx_pred/ mag_norm_pred],axis=-1)
    grad_y = K.concatenate(tensors=[0.0/ mag_norm_pred,
                                    1.0/ mag_norm_pred, dzdy_pred/ mag_norm_pred],axis=-1)


    dot_term_x = K.mean(K.sum(normals[0,:,:,:] * grad_x[0,:,:,:], axis=-1, keepdims=True), axis=-1)
    dot_term_y = K.mean(K.sum(normals[0,:,:,:] * grad_y[0,:,:,:], axis=-1, keepdims=True), axis=-1)


    dot_term_x = K.abs(dot_term_x)
    dot_term_y = K.abs(dot_term_y)

    return K.eval(K.mean(dot_term_x)),K.eval(K.mean(dot_term_y))