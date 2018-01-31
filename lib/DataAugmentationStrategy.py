
import numpy as np


class AbstractDataAugmentationStrategy(object):
    def process_sample(self, features, label, is_test = False):
        if is_test is False:
            aug_features,aug_label = self.process_sample_specialized(features,label)
            return aug_features, aug_label
        else:
            return features, label

    def process_sample_specialized(self, features, label):
        return NotImplemented


class DepthDataAugmentationStrategy(AbstractDataAugmentationStrategy):

    def __init__(self):
        super(DepthDataAugmentationStrategy,self).__init__()
    def process_sample_specialized(self, features, label):

        aug_features = features * np.random.normal(loc=1.0, scale=0.05, size=features.shape)
        #Color shifting
        if (np.random.rand() > 0.85):
            aug_features[:,:,0] =  aug_features[:,:,0]* np.random.normal(loc=1.0, scale=0.1, size=features[:,:,0].shape)
        if (np.random.rand() > 0.85):
            aug_features[:,:,1] =  aug_features[:,:,1]* np.random.normal(loc=1.0, scale=0.1, size=features[:,:,0].shape)
        if (np.random.rand() > 0.85):
            aug_features[:, :, 2] = aug_features[:, :, 2] * np.random.normal(loc=1.0, scale=0.1, size=features[:,:,0].shape)

        return aug_features, label

class HorFlipAugmentationStrategy(AbstractDataAugmentationStrategy):

    def __init__(self, prob):
        self.prob = prob
        super(HorFlipAugmentationStrategy,self).__init__()
    def process_sample_specialized(self, features, label):

        if np.random.uniform() > self.prob:
            aug_f = np.flip(features, axis=1)
            aug_l = np.flip(label, axis=1)
            return aug_f, aug_l
        else:
            return features, label