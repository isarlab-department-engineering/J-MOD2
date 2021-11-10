import numpy as np
from itertools import tee, islice

try:
	# Python 2
	from itertools import izip
except ImportError:
	# Python 3
	izip = zip

class SampleGenerationStrategy(object):
	def __init__(self, sample_type):
		self.sample_type = sample_type

	@staticmethod
	def nwise(iterable, n=2):
		iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
		return izip(*iters)

	def get_image_set(self, sequence):
		sample = []
		for img_set, depth_set, obs_set in zip(self.nwise(sequence.get_image_paths(), 1), self.nwise(sequence.get_labels()[0], 1), self.nwise(sequence.get_labels()[1], 1)):
			label_set = []
			label_set.append(depth_set)
			label_set.append(obs_set)
			sample.append(self.sample_type(img_set, label_set))
		return sample

	def generate_data(self, sequences):
		img_sets = []
		for seq in sequences:
			curr_seq = sequences[seq] #Object Sequence
			seq_set = self.get_image_set(curr_seq)
			img_sets = np.append(img_sets, seq_set)
		return img_sets