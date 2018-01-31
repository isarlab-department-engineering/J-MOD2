import numpy as np
from itertools import tee, islice
from SampleType import ImageDenoisingAESample, RGBSingleChannelDenoisingAESample, DepthDenoisingAESample, FullMAESample
try:
   # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

class SampleGenerationStrategy(object):
    def __init__(self, mean=None):
        self.mean = mean
        self.config = None
        self.width = 0
        self.height = 0

    def generate_data(self, sequences):

        if self.config is not None:
            self.width = self.config.input_width
            self.height = self.config.input_height
        else:
            raise Exception("PASS CONFIG FILE TO GENERATIONSTRATEGY")

        img_sets = []
        for seq in sequences:
            curr_seq = sequences[seq]
            seq_set = self.get_image_set(curr_seq)
            img_sets = np.append(img_sets, seq_set)

        return img_sets

    def get_image_set(self, sequence):

        raise ("Not implemented - this is an abstract method")

    @staticmethod
    def nwise(iterable, n=2):
        iters = (islice(each, i, None) for i, each in enumerate(tee(iterable, n)))
        return izip(*iters)

class PairGenerationStrategy(SampleGenerationStrategy):
    def __init__(self, sample_type):
        super(PairGenerationStrategy, self).__init__()
        self.sample_type = sample_type
    def get_image_set(self, sequence):

        pairs = []
        for img_set, label_set in zip(self.nwise(sequence.get_image_paths(), 2), self.nwise(sequence.get_labels(), 2)):
            pairs.append(self.sample_type(img_set[0], img_set[1], label_set[1], sequence.get_is_grayscale(), mean = self.mean))
        return pairs

class SingleFrameGenerationStrategy(SampleGenerationStrategy):
    def __init__(self,sample_type,get_obstacles=False):
        self.get_obstacles = get_obstacles
        super(SingleFrameGenerationStrategy, self).__init__()
        self.sample_type = sample_type
    def get_image_set(self, sequence):

        sample = []
        if self.get_obstacles:
            for img_set, depth_set, obs_set in zip(self.nwise(sequence.get_image_paths(), 1), self.nwise(sequence.get_labels()[0], 1), self.nwise(sequence.get_labels()[1], 1)):
                label_set = []
                label_set.append(depth_set)
                label_set.append(obs_set)
                sample.append(self.sample_type(img_set, label_set, sequence.get_is_grayscale()))
            return sample
        else:
            for img_set, label_set in zip(self.nwise(sequence.get_image_paths(), 1), self.nwise(sequence.get_labels(), 1)):
                sample.append(self.sample_type(img_set, label_set, sequence.get_is_grayscale(), mean = self.mean))
            return sample

##Generators used for Cadena baseline

class SegmentationAutoEncoderGenerationStrategy(SampleGenerationStrategy):
    def __init__(self, mean = None):
        super(SegmentationAutoEncoderGenerationStrategy, self).__init__(mean)
    def get_image_set(self, sequence):
        pairs = []
        for data_set in sequence.get_image_paths():
            sample = ImageDenoisingAESample(data_set,mean = self.mean)
            sample.set_new_size(self.width, self.height)
            pairs.append(sample)
        return pairs
class RGB_SingleChannel_AutoEncoderGenerationStrategy(SampleGenerationStrategy):
    def __init__(self,  mean = None):
        super(RGB_SingleChannel_AutoEncoderGenerationStrategy, self).__init__(mean)

    def get_image_set(self, sequence):
        if self.config is not None:
            self.channel = self.config.rgb_ae_channel
        else:
            raise("Pass config object to the generator")
        pairs = []
        for data_set in sequence.get_image_paths():
            sample = RGBSingleChannelDenoisingAESample(data_set,self.channel,mean = self.mean)
            sample.set_new_size(self.width,self.height)
            pairs.append(sample)
        return pairs

class DepthAutoEncoderGenerationStrategy(SampleGenerationStrategy):
    def __init__(self, mean = None):
        super(DepthAutoEncoderGenerationStrategy, self).__init__(mean)
    def get_image_set(self, sequence):
        pairs = []
        for data_set in sequence.get_image_paths():
            sample = DepthDenoisingAESample(data_set,mean = self.mean)
            sample.set_new_size(self.width, self.height)
            pairs.append(sample)
        return pairs

class FullMAEGenerationStrategy(SampleGenerationStrategy):
    def get_image_set(self, sequence):
        pairs = []
        for data_set in sequence.get_image_paths():
            sample = FullMAESample(data_set, mean=self.mean)
            sample.set_new_size(self.width, self.height)
            pairs.append(sample)
        return pairs

