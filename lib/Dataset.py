import os
from collections import OrderedDict
from Sequence import Sequence

class Dataset(object):
	def __init__(self, config, data_generation_strategy):
		self.config = config
		self.data_generation_strategy = data_generation_strategy
		self.training_seqs = OrderedDict() # Dictionary of Sequence object
		self.test_seqs = OrderedDict() # Dic of Sequence object

	def read_data(self):
		for dir in self.config.data_train_dirs:
			seq_dir = os.path.join(self.config.data_set_dir, dir)
			self.training_seqs[dir] = Sequence(os.path.join(seq_dir, 'rgb'),
											gt_directory=os.path.join(seq_dir, 'depth'), 
											obstacles_directory=os.path.join(seq_dir,'obstacles_30m'),
											extension=self.config.img_extension)

		for dir in self.config.data_test_dirs:
			seq_dir = os.path.join(self.config.data_set_dir, dir)
			self.training_seqs[dir] = Sequence(os.path.join(seq_dir, 'rgb'),
											gt_directory=os.path.join(seq_dir, 'depth'), 
											obstacles_directory=os.path.join(seq_dir,'obstacles_30m'),
											extension=self.config.img_extension)

	def generate_train_test_data(self):
		train_data = self.data_generation_strategy.generate_data(self.training_seqs)
		test_data = self.data_generation_strategy.generate_data(self.test_seqs)
		return train_data, test_data