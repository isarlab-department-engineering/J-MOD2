import os
from Sequence import AutoEncoderSequence,FullMultiAutoEncoderSequence, PixelwiseSequence, PixelwiseSequenceWithObstacles
from collections import OrderedDict
import numpy as np
import cv2

import os.path


class Dataset(object):

    def __init__(self, config, data_generation_strategy, name):
        self.config = config
        self.name = name
        self.data_generation_strategy = data_generation_strategy
        self.training_seqs = OrderedDict()
        self.test_seqs = OrderedDict()
        self.mean = None
        #self.compute_stats()


    def read_data(self):
        raise Exception("Not implemented - this is an abstract method")

    def print_info(self, show_sequence=False):
        print('--------------------------------------')
        print('------Dataset Info--------')
        print('Dataset Name: {}'.format(self.name))
        print('Number of Training dirs: {}'.format(len(self.training_seqs)))
        print('Training dirs:')
        for directory in self.training_seqs:
            curr_sequence = self.training_seqs[directory]
            print(directory,
                  curr_sequence.sequence_dir,
                  'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                  'Num label: {}'.format(curr_sequence.get_num_label()))
            if show_sequence:
                curr_sequence.visualize_sequence()
        print('Number of Test dirs: {}'.format(len(self.test_seqs)))
        print('Test dirs:')
        for directory in self.test_seqs:
            curr_sequence = self.test_seqs[directory]
            print(directory,
                  curr_sequence.sequence_dir,
                  'Num imgs: {}'.format(curr_sequence.get_num_imgs()),
                  'Num label: {}'.format(curr_sequence.get_num_label()))
            if show_sequence:
                curr_sequence.visualize_sequence()

    def compute_stats(self):
        print("----------------------")

        mean_file = '{}/{}/{}_mean.npy'.format(self.config.data_set_dir, self.config.data_main_dir, self.name)

        if (os.path.isfile(mean_file)):
            self.mean = np.load(mean_file)
            print('Mean loaded from file in {}'.format(mean_file))
        else:
            print('Computing mean for {} dataset....'.format(self.name))
            mean_img = 0
            iter = 0

            for directory in self.training_seqs:
                curr_sequence = self.training_seqs[directory]

                for img_file in curr_sequence.image_paths:
                    curr_img = cv2.imread(img_file)
                    curr_img = np.asarray(curr_img)
                    curr_img = curr_img.astype('float32')

                    mean_img += curr_img
                    iter += 1

            mean_img /= iter
            self.mean = mean_img

            np.save(mean_file, self.mean)

    def get_seq_stat(self):

        raise Exception("Not implemented - this is an abstract method")

    def generate_train_test_data(self):
        train_data = self.data_generation_strategy.generate_data(self.training_seqs)
        test_data = self.data_generation_strategy.generate_data(self.test_seqs)
        return train_data, test_data

    def rescale_and_crop_imgs(self):
         for sequence in self.training_seqs:
             self.training_seqs[sequence].resize_and_rescale(self.config.input_height,
                                                             self.config.input_width)
        # for sequence in self.test_seqs:
        #     self.test_seqs[sequence].resize_and_rescale(self.config.input_height,
        #                                                     self.config.input_width)

class UnrealDataset(Dataset):
    def __init__(self, config, data_generation_strategy, name):
        print('--------------------------------------')
        print('------Processing UnrealDataset Dataset--------')
        print('--------------------------------------')
        super(UnrealDataset, self).__init__(config, data_generation_strategy, name)


    def get_seq_stat(self):
        num_tr_seq = len(self.training_seqs)
        num_te_seq = len(self.test_seqs)
        num_tr_imgs = 0
        num_te_imgs = 0
        for seq in self.training_seqs:
            num_tr_imgs += self.training_seqs[seq].get_num_imgs()
        for seq in self.test_seqs:
            num_te_imgs += self.test_seqs[seq].get_num_imgs()
        return num_tr_seq, num_te_seq, num_tr_imgs, num_te_imgs

class UnrealDataset_DepthSupervised(UnrealDataset):

    def __init__(self, config, data_generation_strategy, read_obstacles = False):
        super(UnrealDataset_DepthSupervised, self).__init__(config, data_generation_strategy, 'UnrealDataset')
        self.read_obstacles = read_obstacles

    def read_data(self):

        if self.read_obstacles:
            for dir in self.config.data_train_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.training_seqs[dir] = PixelwiseSequenceWithObstacles(os.path.join(seq_dir, 'rgb'),
                                                            self.config.img_extension,
                                                            gt_directory=os.path.join(seq_dir, 'depth'),
                                                            obstacles_directory=os.path.join(seq_dir,'obstacles_20m'),
                                                            is_grayscale=False,
                                                            name='ud_train/' + dir)

            for dir in self.config.data_test_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.test_seqs[dir] = PixelwiseSequenceWithObstacles(os.path.join(seq_dir, 'rgb'),
                                                        self.config.img_extension,
                                                        gt_directory=os.path.join(seq_dir, 'depth'),
                                                        obstacles_directory=os.path.join(seq_dir, 'obstacles_20m'),
                                                        is_grayscale=False,
                                                        name='ud_test/' + dir)

        else:
            for dir in self.config.data_train_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.training_seqs[dir] = PixelwiseSequence(os.path.join(seq_dir, 'rgb'),
                                                   self.config.img_extension,
                                                   gt_directory=os.path.join(seq_dir, 'depth'),
                                                   is_grayscale=False,
                                                   name='ud_train/' + dir)

            for dir in self.config.data_test_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.test_seqs[dir] = PixelwiseSequence(os.path.join(seq_dir, 'rgb'),
                                               self.config.img_extension,
                                               gt_directory=os.path.join(seq_dir, 'depth'),
                                               is_grayscale=False,
                                               name='ud_test/' + dir)

class UnrealDataset_Autoencoder(UnrealDataset):

    def __init__(self, config, data_subset, data_generation_strategy, name='UnrealDataset'):
        super(UnrealDataset_Autoencoder, self).__init__(config, data_generation_strategy, name)
        self.data_subset = data_subset

    def read_data(self):
        if self.data_subset is not 'full':
            for dir in self.config.data_train_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.training_seqs[dir] = AutoEncoderSequence(os.path.join(seq_dir, self.data_subset),
                                                              self.config.img_extension,
                                                              is_grayscale=False,
                                                              name='ud_train/' + dir)

            for dir in self.config.data_test_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.test_seqs[dir] = AutoEncoderSequence(os.path.join(seq_dir, self.data_subset),
                                                          self.config.img_extension,
                                                          is_grayscale=False,
                                                          name='ud_train/' + dir)
        else:
            for dir in self.config.data_train_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.training_seqs[dir] = FullMultiAutoEncoderSequence(os.path.join(seq_dir),
                                                                       self.config.img_extension,
                                                                       is_grayscale=False,
                                                                       name='ud_train/' + dir)

            for dir in self.config.data_test_dirs:
                seq_dir = os.path.join(self.config.data_set_dir, self.config.data_main_dir, dir)
                self.test_seqs[dir] = FullMultiAutoEncoderSequence(os.path.join(seq_dir),
                                                                   self.config.img_extension,
                                                                   is_grayscale=False,
                                                                   name='ud_train/' + dir)
