import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Sequence(object):

    def __init__(self, directory, extension, label_file, is_grayscale = False, name='Sequence'):

        self.sequence_name = name
        self.sequence_dir = directory
        self.is_grayscale = is_grayscale
        self.image_paths = []
        self.label = []

        self.load_img_paths(extension)
        self.load_label(label_file)

    def load_img_paths(self, extension):
        self.image_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + extension)))

    def load_label(self, label_file):
        self.label = np.loadtxt(os.path.join(self.sequence_dir, label_file))

    def get_num_imgs(self):
        return len(self.image_paths)

    def get_num_label(self):
        return len(self.label)

    def get_image_paths(self):
        return self.image_paths

    def get_labels(self):
        return self.label

    def get_is_grayscale(self):
        return self.is_grayscale

    def resize_and_rescale(self, height, width):
        sub_sampled_dir = os.path.join(self.sequence_dir, 'downsampled')
        if not os.path.exists(os.path.join(sub_sampled_dir)):
            os.makedirs(sub_sampled_dir)
        for img_file in self.image_paths:
            curr_img = cv2.imread(img_file)
            curr_img = cv2.resize(curr_img,  (width, height), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(sub_sampled_dir, os.path.basename(img_file)), curr_img)
            print('Processing img: {}'.format(img_file))

    def visualize_sequence(self):
        plt.figure(1)
        plt.title(self.sequence_name)
        for img in self.image_paths:
            print('Testing img: ', img)
            plt.clf()
            plt.imshow(cv2.imread(img))
            plt.pause(0.001)

        plt.close()

class PixelwiseSequence(object):
    def __init__(self, input_directory, extension, gt_directory, is_grayscale = False, name='Sequence'):

        self.sequence_name = name
        self.sequence_dir = input_directory
        self.gt_dir = gt_directory
        self.is_grayscale = is_grayscale
        self.image_paths = []
        self.label_paths = []

        self.load_img_paths(extension)
        self.load_label_paths(extension)

    def load_img_paths(self, extension):
        self.image_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + extension)))

    def load_label_paths(self, extension):
        self.label_paths = sorted(glob(os.path.join(self.gt_dir, '*' + '.' + extension)))

    def get_num_imgs(self):
        return len(self.image_paths)

    def get_num_label(self):
        return len(self.label_paths)

    def get_image_paths(self):
        return self.image_paths

    def get_labels(self):
        return self.label_paths

    def get_is_grayscale(self):
        return self.is_grayscale

    def resize_and_rescale(self, height, width):
        sub_sampled_dir = os.path.join(self.sequence_dir, 'downsampled')

        if not os.path.exists(os.path.join(sub_sampled_dir)):
            os.makedirs(sub_sampled_dir)
        for img_file in self.image_paths:
            curr_img = cv2.imread(img_file)
            curr_img = cv2.resize(curr_img,  (width, height), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(sub_sampled_dir, os.path.basename(img_file)), curr_img)
            print('Processing img: {}'.format(img_file))

    def visualize_sequence(self):
        plt.figure(1)
        plt.title(self.sequence_name)
        for img in self.image_paths:
            print('Testing img: ', img)
            plt.clf()
            plt.imshow(cv2.imread(img))
            plt.pause(0.001)
        for label in self.label_paths:
            print('Testing label: ', label)
            plt.clf()
            plt.imshow(cv2.imread(label))
            plt.pause(0.001)

        plt.close()

class PixelwiseSequenceWithObstacles(PixelwiseSequence):
    def __init__(self, input_directory, extension, gt_directory, obstacles_directory, is_grayscale=False, name='Sequence'):
        self.obstacles_dir = obstacles_directory
        self.load_obstacles_paths(extension='txt')
        super(PixelwiseSequenceWithObstacles, self).__init__(input_directory, extension, gt_directory, is_grayscale, name)
    def load_obstacles_paths(self, extension):
        self.obstacles_paths = sorted(glob(os.path.join(self.obstacles_dir, '*' + '.' + extension)))
    def get_labels(self):
        return (self.label_paths, self.obstacles_paths)

class MultiDataSequence(object):

    def __init__(self, directories, extension, gt_dir, name='Sequence'):

        self.sequence_name = name
        self.sequence_dir = directories
        self.image_paths = []
        self.gt_paths = []
        self.extension = extension
        self.gt_dir = gt_dir

        self.load_img_paths()
        self.load_label()

    def load_img_paths(self):
        for dir in self.sequence_dir:
            self.image_paths.append(sorted(glob(os.path.join(dir, '*' + '.' + self.extension))))

    def load_label(self):

        self.gt_paths = sorted(glob(os.path.join(self.gt_dir, '*' + '.' + self.extension)))

    def get_num_imgs(self):
        num_images = 0
        for seq in self.image_paths:
            num_images += len(seq)
        return num_images/len(self.image_paths)

    def get_num_label(self):
        return len(self.gt_paths)

    def get_image_paths(self):
        return self.image_paths

    def get_labels(self):
        return self.gt_paths

    def resize_and_rescale(self, height, width):
        raise ("TODO - to implement")

    def visualize_sequence(self):
        plt.figure(1)
        plt.title(self.sequence_name)
        for seq in self.image_paths:
            for img in seq:
                print('Testing img: ', img)
                plt.clf()
                plt.imshow(cv2.imread(img))
                plt.pause(0.001)

        plt.close()

#Used by Cadena baseline
class AutoEncoderSequence(object):
    def __init__(self, directory, extension, is_grayscale = False, name='Sequence'):
        self.sequence_name = name
        self.sequence_dir = directory
        self.is_grayscale = is_grayscale

        self.data_paths = []
        self.load_paths(extension)

    def load_paths(self,extension):
        self.data_paths = sorted(glob(os.path.join(self.sequence_dir, '*' + '.' + extension)))
    def get_len_data(self):
        return len(self.data_paths)
    def get_image_paths(self):
        return self.data_paths
    #uniformity with other non-autoencoder sequences
    def get_label_paths(self):
        return self.get_image_paths()
    def get_is_grayscale(self):
        return self.is_grayscale
    def get_num_imgs(self):
        return self.get_len_data()
    def get_num_label(self):
        return self.get_num_imgs()

class FullMultiAutoEncoderSequence(object):
    def __init__(self, directory, extension, is_grayscale=False, name='Sequence'):
        self.sequence_name = name
        self.sequence_dir = directory


        self.is_grayscale = is_grayscale

        self.d_sequence = AutoEncoderSequence(os.path.join(directory, 'depth'), extension, is_grayscale, name ='Depth')
        self.rgb_sequence = AutoEncoderSequence(os.path.join(directory, 'rgb'), extension, is_grayscale, name ='RGB')
        self.s_sequence = AutoEncoderSequence(os.path.join(directory,'segmentation2'),extension,is_grayscale, name = 'Segmentation')

        self.d_paths = self.d_sequence.data_paths
        self.rgb_paths = self.rgb_sequence.data_paths
        self.seg_paths = self.s_sequence.data_paths

        self.data_paths = []
        self.load_paths(extension)

    def load_paths(self,extension):
        for i in range(0, len(self.d_paths)):
            self.data_paths.append([self.d_paths[i], self.rgb_paths[i], self.seg_paths[i]])
    def get_len_data(self):
        return len(self.d_paths)
    def get_image_paths(self):
        return self.data_paths
    #uniformity with other non-autoencoder sequences
    def get_label_paths(self):
        return self.get_image_paths()
    def get_is_grayscale(self):
        return self.is_grayscale
    def get_num_imgs(self):
        return self.get_len_data()
    def get_num_label(self):
        return self.get_num_imgs()


