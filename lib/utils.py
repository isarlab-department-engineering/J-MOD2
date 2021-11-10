import os
import cv2
from datetime import datetime

def get_time():
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def prepare_dirs(config):
	config.model_name = "{}_{}_{}_{}_test_dirs_{}_{}".format(config.exp_name,
															 config.data_main_dir,
															 config.input_height,
															 config.input_width,
															 config.data_test_dirs,
															 get_time())
	config.model_dir = os.path.join(config.log_dir, config.model_name)
	config.tensorboard_dir = os.path.join(config.log_dir, config.model_name, 'tensorboard')
	for path in [config.log_dir, config.model_dir, config.tensorboard_dir]:
		if not os.path.exists(path):
			if config.is_train:
				os.makedirs(path)