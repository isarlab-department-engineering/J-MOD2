from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.vgg19 import VGG19

from lib.SampleType import Depth_SingleFrame
import keras.backend as K
from lib.DataAugmentationStrategy import DepthDataAugmentationStrategy


import numpy as np

from lib.Dataset import UnrealDataset
from AbstractModel import AbstractModel
from lib.DataGenerationStrategy import SingleFrameGenerationStrategy


class FullyConvolutionalModel(AbstractModel):

    def __init__(self, config):

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory_fraction
        tf_config.gpu_options.visible_device_list = "0"
        set_session(tf.Session(config=tf_config))

        self.is_deploy = config.is_deploy
        super(FullyConvolutionalModel, self).__init__(config)

    def prepare_data(self):

        if not self.is_deploy:
            dataset, dataset_name = self.load_dataset()
            dataset.data_generation_strategy.config = self.config

            self.dataset[dataset_name] = dataset
            self.dataset[dataset_name].read_data()
            self.dataset[dataset_name].print_info(show_sequence=False)

            self.data_augmentation_strategy = DepthDataAugmentationStrategy()

            self.training_set, self.test_set = self.dataset[dataset_name].generate_train_test_data()

            num_tr_seq, num_te_seq, num_tr_imgs, num_te_imgs = self.dataset[dataset_name].get_seq_stat()

            expected_train_img_pair = num_tr_imgs
            expected_test_img_pair = num_te_imgs

            print('Number of training pairs: {}'.format(len(self.training_set)))
            print('Number of expected training pairs: {}'.format(expected_train_img_pair))
            assert len(self.training_set) == expected_train_img_pair, \
                "Num of computed train pairs does not correspond to the expected one"

            print("Splitting train and validation...")
            np.random.shuffle(self.training_set)
            train_val_split_idx = int(len(self.training_set)*(1-self.config.validation_split))
            self.validation_set = self.training_set[train_val_split_idx:]
            self.training_set = self.training_set[0:train_val_split_idx]

            print('Number of training pairs (no validation): {}'.format(len(self.training_set)))
            print('Number of validation pairs: {}'.format(len(self.validation_set)))

            print('Number of test pairs: {}'.format(len(self.test_set)))
            assert len(self.test_set) == expected_test_img_pair,\
                "Num of computed test pairs does not correspond to the expected one"

