from keras.callbacks import Callback
import tensorflow as tf
import os
import numpy as np
import glob
import keras.backend as K
from pkg_resources import parse_version
import math
from keras.models import Model

class TensorBoardCustom(Callback):
	def __init__(self, config, x_test, y_test, log_dir='./logs', histogram_freq=20, write_graph=True, write_images=False):
		super(TensorBoardCustom, self).__init__()
		if K.backend() != 'tensorflow':
			raise RuntimeError('TensorBoard callback only works '
								'with the TensorFlow backend.')
		self.log_dir = log_dir
		self.config = config
		self.histogram_freq = histogram_freq
		self.merged = None
		self.write_graph = write_graph
		self.write_images = write_images
		self.X_test = x_test
		self.Y_test = y_test
		self.global_step = 0
		self.model = []
		self.sess = []

	def set_model(self, model):
		self.model = model
		self.sess = K.get_session()
		if self.histogram_freq and self.merged is None:
			for layer in self.model.layers:
				if (layer.name=='block1_conv1'):
					print(np.shape(layer.input[:, :, :, 0]))
					abs_img = tf.expand_dims(layer.input[:, :, :, 0], axis=-1)
					phase_img = tf.expand_dims(layer.input[:, :, :, 1], axis=-1)
					tf.summary.image('{}_in_abs'.format(layer.name),
									 abs_img, max_outputs=self.config.max_image_summary)
					tf.summary.image('{}_in_phase'.format(layer.name), phase_img, max_outputs=self.config.max_image_summary)
				if (layer.name=='depth_output'):
					tf.summary.image('{}_estimated_dem'.format(layer.name),
									 layer.output, max_outputs=self.config.max_image_summary)
					tf.summary.image('{}_gt_dem'.format(layer.name),
									 self.Y_test[0], max_outputs=self.config.max_image_summary)
		self.merged = tf.summary.merge_all()
		if self.write_graph:
			self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
		else:
			self.writer = tf.summary.FileWriter(self.log_dir)

	def on_batch_end(self, batch, logs=None):
		logs = logs or {}
		if batch % self.histogram_freq == 0:
			if self.model.uses_learning_phase:
				try:
					cut_v_data = self.model.input.shape[0]
					val_data = self.X_test[:cut_v_data] + [0]
					tensors = self.model.input + [K.learning_phase()]
				except:
					print "Tensor batch size not defined. Skipping"
					return
			else:
				val_data = self.X_test
				tensors = self.model.input
			feed_dict = {tensors: val_data}
			result = self.sess.run([self.merged], feed_dict=feed_dict)
			summary_str = result[0]
			self.writer.add_summary(summary_str, self.global_step)
		for name, value in logs.items():
			if name in ['batch', 'size']:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, self.global_step)
		self.writer.flush()
		self.global_step = self.global_step + 1

	def on_train_end(self, _):
		self.writer.close()

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		for name, value in logs.items():
			if name in ['batch', 'size', 'loss', 'rmse_metric']:
				continue
			summary = tf.Summary()
			summary_value = summary.value.add()
			summary_value.simple_value = value.item()
			summary_value.tag = name
			self.writer.add_summary(summary, epoch)
		self.writer.flush()

class PrintBatch(Callback):
	def on_batch_end(self, batch, logs={}):
		print(logs)