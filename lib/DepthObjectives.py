import tensorflow as tf
import keras.backend as K
import numpy as np
from config import get_config

def log_normals_loss(y_true, y_pred):
	# Get batch sie
	config, unparsed = get_config()
	batch_size = config.batch_size
	# Print batch
	y_true = tf.Print(y_true, [y_true], message='y_true', summarize=30)
	y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=30)
	y_true_clipped = y_true
	y_pred_clipped = y_pred
	# aux filter
	w_x = K.variable(np.array([[-1.0, 0.0, 1.0],
							   [-1.0, 0.0, 1.0],
							   [-1.0, 0.0, 1.0]]).reshape(3, 3, 1, 1))
	w_y = K.variable(np.array([[-1.0, -1.0, -1.0],
								[0.0, 0.0, 0.0],
								[1.0, 1.0, 1.0]]).reshape(3, 3, 1, 1))
	# true
	dzdx = K.conv2d(y_true_clipped, w_x, padding='same')
	dzdy = K.conv2d(y_true_clipped, w_y, padding='same')
	dzdx_ = dzdx * -1.0
	dzdy_ = dzdy * -1.0
	mag_norm = K.pow(dzdx,2) + K.pow(dzdy,2) + 1.0
	mag_norm = K.sqrt(mag_norm)
	# Normals
	N3 = 1.0 / mag_norm
	N1 = dzdx_ / mag_norm
	N2 = dzdy_ / mag_norm
	normals = K.concatenate(tensors=[N1,N2,N3],axis=-1)
	# pred
	dzdx_pred = K.conv2d(y_pred_clipped, w_x, padding='same')
	dzdy_pred = K.conv2d(y_pred_clipped, w_y, padding='same')
	mag_norm_pred_x = K.pow(dzdx_pred,2) + 1.0
	mag_norm_pred_x = K.sqrt(mag_norm_pred_x)
	mag_norm_pred_y = K.pow(dzdy_pred, 2) + 1.0
	mag_norm_pred_y = K.sqrt(mag_norm_pred_y)
	#
	grad_x = K.concatenate(tensors=[K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x,
									K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_x, dzdx_pred/ mag_norm_pred_x],axis=-1)
	grad_y = K.concatenate(tensors=[K.constant(0.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y,
									K.constant(1.0, shape=[batch_size, K.int_shape(y_pred)[1], K.int_shape(y_pred)[2], K.int_shape(y_pred)[3]])/ mag_norm_pred_y, dzdy_pred/ mag_norm_pred_y],axis=-1)
	# compute d_i
	first_log = K.log(y_pred_clipped + 1.)
	second_log = K.log(y_true_clipped + 1.)
	log_term = K.mean(K.square(first_log - second_log), axis=-1)
	# dot prod
	dot_term_x = K.sum(normals[:,:,:,:] * grad_x[:,:,:,:], axis=-1, keepdims=True)
	dot_term_y = K.sum(normals[:,:,:,:] * grad_y[:,:,:,:], axis=-1, keepdims=True)
	dot_term_x = tf.Print(dot_term_x, [dot_term_x], message='dot_term_x', summarize=30)
	dot_term_y = tf.Print(dot_term_y, [dot_term_y], message='dot_term_y', summarize=30)
	# second term
	sc_inv_term = K.square(K.mean((first_log - second_log), axis=-1))
	# first term + dy term
	norm_term = K.mean(K.square(dot_term_x), axis=-1) + K.mean(K.square(dot_term_y), axis=-1)
	diff_x = dzdx_pred - dzdx
	diff_y = dzdy_pred - dzdy
	grad_loss = K.mean(K.square(diff_x) + K.square(diff_y), axis=-1)
	loss = log_term - (0.5 * sc_inv_term) + norm_term
	return loss
