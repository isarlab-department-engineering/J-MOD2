import os
from glob import glob

class Sequence(object):
	def __init__(self, input_directory, gt_directory, obstacles_directory, extension, is_grayscale=True):
		self.sequence_dir = input_directory
		self.gt_dir = gt_directory
		self.obstacles_dir = obstacles_directory
		self.extension = extension
		self.input_paths = []
		self.gt_paths =[]
		self.obstacles_paths = []
		# Init
		self.load_input_paths()
		self.load_gt_paths()
		self.load_obstacles_paths()

	def load_input_paths(self):
		self.input_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + self.extension)))

	def load_gt_paths(self):
		self.gt_paths = sorted(glob(os.path.join(self.gt_dir, '*' + '.' + self.extension)))

	def load_obstacles_paths(self):
		self.obstacles_paths = sorted(glob(os.path.join(self.obstacles_dir, '*' + '.' + 'txt')))

	def get_image_paths(self):
		return self.input_paths

	def get_labels(self):
		return (self.gt_paths, self.obstacles_paths)