import tensorflow as tf
import keras.backend as K
import numpy as np
from config import get_config


def root_mean_squared_logarithmic_loss(y_true, y_pred):
    y_true = tf.Print(y_true, [y_true], message='y_true', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=30)
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)

    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1)+0.00001)


def root_mean_squared_loss(y_true, y_pred):
    #y_true = tf.Print(y_true, [y_true], message='y_true-rmse', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred[0, :, :]], message='y_pred-rmse1', summarize=30)
    y_pred = tf.Print(y_pred, [y_true[0, :, :]], message='y_true-rmse1', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred[1, :, :]], message='y_pred-rmse2', summarize=30)
    y_pred = tf.Print(y_pred, [y_true[1, :, :]], message='y_true-rmse2', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred[30, :, :]], message='y_pred-rmse3', summarize=30)
    y_pred = tf.Print(y_pred, [y_true[30, :, :]], message='y_true-rmse3', summarize=30)
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)+0.00001)
    #return K.sqrt(K.square(y_pred - y_true) + 0.00001)


def mean_squared_loss(y_true, y_pred):
    y_true = tf.Print(y_true, [y_true], message='y_true-mse', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred], message='y_pred-mse', summarize=30)
    mse = K.square(y_pred - y_true)
    mse = tf.Print(mse, [mse], message='mse error', summarize=30)
    return mse
def eigen_loss(y_true, y_pred):
    y_true = tf.Print(y_true, [y_true], message='y_true', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=30)

    y_true_clipped = K.clip(y_true, K.epsilon(), None)
    y_pred_clipped = K.clip(y_pred, K.epsilon(), None)

    first_log = K.log(y_pred_clipped + 1.)
    second_log = K.log(y_true_clipped + 1.)
    w_x = K.variable(np.array([[-1., 0., 1.],
                                [-1., 0., 1.],
                                [-1., 0., 1.]]).reshape(3, 3, 1, 1))

    grad_x_pred = K.conv2d(first_log, w_x, padding='same')
    grad_x_true = K.conv2d(second_log, w_x, padding='same')

    w_y = K.variable(np.array([[-1., -1., -1.],
                                [0., 0., 0.],
                                [1., 1., 1.]]).reshape(3, 3, 1, 1))

    grad_y_pred = K.conv2d(first_log, w_y, padding='same')
    grad_y_true = K.conv2d(second_log, w_y, padding='same')
    diff_x = grad_x_pred - grad_x_true
    diff_y = grad_y_pred - grad_y_true

    log_term = K.mean(K.square((first_log - second_log)), axis=-1)
    sc_inv_term = K.square(K.mean((first_log - second_log),axis=-1))
    grad_loss = K.mean(K.square(diff_x) + K.square(diff_y), axis=-1)

    return log_term - (0.5 * sc_inv_term) + grad_loss

def log_normals_loss(y_true, y_pred):
    y_true = tf.Print(y_true, [y_true], message='y_true', summarize=30)
    y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=30)

    #compute normals with convolution approach
    # (http://answers.opencv.org/question/82453/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-product/)
    config, unparsed = get_config()
    batch_size = config.batch_size

    #y_true_clipped = K.clip(y_true, K.epsilon(), None)#40.0)
    #y_pred_clipped = K.clip(y_pred, K.epsilon(), None)
    y_true_clipped = y_true
    y_pred_clipped = y_pred

    filter_y = K.variable(np.array([[ 0., -0.5 , 0.],
                               [0., 0., 0.],
                               [0., 0.5, 0.]]).reshape(3, 3, 1, 1))


    filter_x = K.variable(np.array([ [0, 0., 0.],
                               [0.5, 0., -0.5],
                               [0., 0., 0.]]).reshape(3, 3, 1, 1))
    w_x = K.variable(np.array([[-1., 0., 1.],
                               [-1., 0., 1.],
                               [-1., 0., 1.]]).reshape(3, 3, 1, 1))

    w_y = K.variable(np.array([[-1., -1., -1.],
                               [0., 0., 0.],
                               [1., 1., 1.]]).reshape(3, 3, 1, 1))

    #dzdx = K.conv2d(K.exp(y_true_clipped), w_x, padding='same')
    #dzdy = K.conv2d(K.exp(y_true_clipped), w_y, padding='same')
    dzdx = K.conv2d(y_true_clipped, w_x, padding='same')
    dzdy = K.conv2d(y_true_clipped, w_y, padding='same')

    dzdx_ = dzdx * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdx))
    dzdy_ = dzdy * -1.0#K.constant(-1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(-1.0, shape=K.int_shape(dzdy))

    mag_norm = K.pow(dzdx,2) + K.pow(dzdy,2) + 1.0#K.constant(1.0, shape=[batch_size,K.int_shape(y_pred)[1],K.int_shape(y_pred)[2],K.int_shape(y_pred)[3]]) #K.constant(1.0, shape=K.int_shape(dzdx))

    mag_norm = K.sqrt(mag_norm)
    N3 = 1.0 / mag_norm #K.constant(1.0, shape=K.int_shape(dzdx)) / mag_norm
    N1 = dzdx_ / mag_norm
    N2 = dzdy_ / mag_norm

    normals = K.concatenate(tensors=[N1,N2,N3],axis=-1)

    #dzdx_pred = K.conv2d(K.exp(y_pred_clipped), w_x, padding='same')
    #dzdy_pred = K.conv2d(K.exp(y_pred_clipped), w_y, padding='same')
    dzdx_pred = K.conv2d(y_pred_clipped, w_x, padding='same')
    dzdy_pred = K.conv2d(y_pred_clipped, w_y, padding='same')

    mag_norm_pred_x = K.pow(dzdx_pred,2) + 1.0
    mag_norm_pred_x = K.sqrt(mag_norm_pred_x)
    mag_norm_pred_y = K.pow(dzdy_pred, 2) + 1.0
    mag_norm_pred_y = K.sqrt(mag_norm_pred_y)

    grad_x = K.concatenate(tensors=[K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x,
                                    K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x, dzdx_pred/ mag_norm_pred_x],axis=-1)
    grad_y = K.concatenate(tensors=[K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y,
                                    K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y, dzdy_pred/ mag_norm_pred_y],axis=-1)

    first_log = K.log(y_pred_clipped + 1.)
    second_log = K.log(y_true_clipped + 1.)

    log_term = K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) + 0.00001)

    dot_term_x = K.sum(normals[:,:,:,:] * grad_x[:,:,:,:], axis=-1, keepdims=True)
    dot_term_y = K.sum(normals[:,:,:,:] * grad_y[:,:,:,:], axis=-1, keepdims=True)
    #dot_term_x = K.mean(K.sum(normals[:, :, :, :] * grad_x[:, :, :, :], axis=-1, keepdims=True), axis=-1)
    #dot_term_y = K.mean(K.sum(normals[:, :, :, :] * grad_y[:, :, :, :], axis=-1, keepdims=True), axis=-1)


    dot_term_x = tf.Print(dot_term_x, [dot_term_x], message='dot_term_x', summarize=30)
    dot_term_y = tf.Print(dot_term_y, [dot_term_y], message='dot_term_y', summarize=30)

    #commentare per vecchia versione
    sc_inv_term = K.square(K.mean((first_log - second_log), axis=-1))
    norm_term = K.mean(K.square(dot_term_x), axis=-1) + K.mean(K.square(dot_term_y), axis=-1)

    diff_x = dzdx_pred - dzdx
    diff_y = dzdy_pred - dzdy
    grad_loss = K.mean(K.square(diff_x) + K.square(diff_y), axis=-1)

    loss = log_term - (0.5 * sc_inv_term) + norm_term #+ grad_loss
    #loss = log_term + K.square(dot_term_x) + K.square(dot_term_y)

    return loss

