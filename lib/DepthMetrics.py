import keras.backend as K

def rmse_metric(y_true, y_pred):
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