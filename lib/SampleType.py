import numpy as np
import cv2
import scipy.io as sio
from scipy import ndimage
import random

class AbstractSample(object):

    def read_features(self):

        raise ("Not implemented - this is an abstract method")

    def read_labels(self):

        raise ("Not implemented - this is an abstract method")

class Depth_SingleFrame(AbstractSample):
    def __init__(self, img, label, is_grayscale = False, mean = None):
        self.img_path = img
        self.depth_gt = label
        self.is_grayscale = is_grayscale
        self.mean = mean
        super(Depth_SingleFrame, self).__init__()

    def read_features(self):
        if self.is_grayscale:
            img = cv2.imread(self.img_path[0], cv2.IMREAD_GRAYSCALE)
            img -= self.mean
            return img
        else:
            img = cv2.imread(self.img_path[0], cv2.IMREAD_COLOR)
            return img

    def read_labels(self):
        label = cv2.imread(self.depth_gt[0], cv2.IMREAD_GRAYSCALE)

        return np.expand_dims(label,2)

class DepthObstacles_SingleFrame(AbstractSample):
    def __init__(self, img, label, is_grayscale = False, mean = None):
        self.img_path = img
        self.depth_gt = label[0]
        self.obstacles_gt = label[1]
        self.is_grayscale = is_grayscale
        self.mean = mean

        super(DepthObstacles_SingleFrame, self).__init__()

    def read_features(self):
        if self.is_grayscale:
            img = cv2.imread(self.img_path[0], cv2.IMREAD_GRAYSCALE)
            img -= self.mean
            return img
        else:
            img = cv2.imread(self.img_path[0], cv2.IMREAD_COLOR)
            return img

    def read_labels(self):
        depth_label = cv2.imread(self.depth_gt[0], cv2.IMREAD_GRAYSCALE)

        #leggi file di testo

        with open(self.obstacles_gt[0],'r') as f:
            obstacles = f.readlines()
        obstacles = [x.strip() for x in obstacles]

        obstacles_label = np.zeros(shape=(5,8,7))

        for obs in obstacles:
            parsed_str_obs = obs.split()
            parsed_obs = np.zeros(shape=(8))
            i = 0
            for n in parsed_str_obs:
                if i < 2:
                    parsed_obs[i] = int(n)
                else:
                    parsed_obs[i] = float(n)
                i += 1

            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 0] = 1.0 #confidence
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 1] = parsed_obs[2] # x
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 2] = parsed_obs[3] # y
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 3] = parsed_obs[4] # w
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 4] = parsed_obs[5] # h
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 5] = parsed_obs[6] * 0.1 # m
            obstacles_label[int(parsed_obs[1]), int(parsed_obs[0]), 6] = parsed_obs[7] * 0.1 # v
        labels = {}
        labels["depth"] = np.expand_dims(depth_label,2)
        labels["obstacles"] =np.reshape(obstacles_label,(40,7))

        return labels

#Samples used by Cadena baseline
class GenericDenoisingAESample(AbstractSample):
    def __init__(self, data, corrupt_data = True, mean = None):
        self.data_path = data
        self.width = 64 #we trained this method at this resolution.  However, this number it edited later accordingly to the config file with a call to set_new_size.
        self.height = 40
        self.corrupt_data = corrupt_data

        self.preload = False

        self.mean = mean

        if self.preload:
            self.features = self.init_features()
            self.label = self.init_labels()

        super(AbstractSample, self).__init__()

    def set_new_size(self,width,height):
        self.width = width
        self.height = height

    def resize_and_expand(self, data):

        data = cv2.resize(data, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if len(data.shape) < 3:
            data = np.expand_dims(data, 2)
        return data

    def init_features(self):
        #print "Loading features in RAM...Please wait"
        data = self.load_data(self.data_path)

        data = self.resize_and_expand(data)
        # data colud be Non when building the graph
        if data is not None and self.corrupt_data:
            corrupted_data = self.apply_noise_routine(data)
            return corrupted_data
        elif data is None:
            return None
        elif not self.corrupt_data:
            return data
    def init_labels(self):
        #print "Loading labels in RAM...Please wait"
        data = self.load_data(self.data_path)
        data = self.resize_and_expand(data)
        return data

    def read_features(self):
        if self.preload:
            return self.features
        else:
            data = self.load_data(self.data_path)

            data = self.resize_and_expand(data)

            if data is not None:
                corrupted_data = self.apply_noise_routine(data)
                return corrupted_data
            else:
                return None

    def read_labels(self):
        if self.preload:
            return self.label
        else:
            data = self.load_data(self.data_path)
            data = self.resize_and_expand(data)
            return data

    def load_data(self, data_path):
        raise ("Not implemented - this is an abstract method")
    def apply_noise_routine(self, data):
        raise ("Not implemented - this is an abstract method")
    def subtract_mean_if_available(self, data, mean):
        if mean is not None:
            return data - mean
        else:
            return data


class ImageDenoisingAESample(GenericDenoisingAESample):
    def __init__(self, data, is_grayscale = False, corrupt_data = True, mean = None):

        self.is_grayscale = is_grayscale

        super(ImageDenoisingAESample, self).__init__(data, corrupt_data=corrupt_data, mean=mean)
    def load_data(self, data_path):

        img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        #if self.mean is not None:
        #    img = self.subtract_mean_if_available(img, self.mean)
        return img

    def apply_noise_routine(self, data):
        #Force about 10% of pixels to zero

        r = 0.1

        final_mask = np.random.choice([0,1],data.shape,p=[r,1-r])

        return final_mask*data

class RGBSingleChannelDenoisingAESample(ImageDenoisingAESample):
    def __init__(self, data, channel, corrupt_data = True,mean = None):
        #0 :B 1:G 2:R
        self.channel = channel
        super(RGBSingleChannelDenoisingAESample, self).__init__(data, corrupt_data=corrupt_data, is_grayscale=True, mean=mean)

    def load_data(self, data_path):
        img = cv2.imread(data_path, cv2.IMREAD_COLOR)[:,:,self.channel]/255.
        #if self.mean is not None:
        #    img = self.subtract_mean_if_available(img, self.mean[:,:,self.channel])
        return img

class DepthDenoisingAESample(ImageDenoisingAESample):
    def __init__(self, data, corrupt_data = True, mean=None):
        super(DepthDenoisingAESample, self).__init__(data, corrupt_data=corrupt_data, is_grayscale=True, mean=mean)

    def load_data(self, data_path):

        depth = cv2.imread(self.data_path, cv2.IMREAD_GRAYSCALE)
        if self.mean is not None:
            depth = self.subtract_mean_if_available(depth, self.mean[:,:,0])
        depth = self.convert_depth_to_meters(depth) #TODO: this should be at an higher lever, as this conversion is required by the UnrealDataset only. For JMOD2, it was done at the model's 'prepare_data_for_model' method
        return depth

    def convert_depth_to_meters(self, depth):
        depth = np.asarray(depth)
        depth = depth.astype('float32')
        # depth = depth * 39.75 / 255.0
        depth = -4.586e-09 * (depth ** 4) + 3.382e-06 * (depth ** 3) - 0.000105 * (
        depth ** 2) + 0.04239 * depth + 0.04072
        depth /= 39.75

        return depth

class FullMAESample(AbstractSample):
    def __init__(self, data, is_grayscale=False, mean=None):

        d_path = data[0]
        rgb_path = data[1]
        s_path = data[2]

        self.d_sample = DepthDenoisingAESample(d_path, mean = mean[0])
        self.r_sample = RGBSingleChannelDenoisingAESample(rgb_path, channel=2, mean= mean[1])
        self.g_sample = RGBSingleChannelDenoisingAESample(rgb_path, channel=1, mean= mean[2])
        self.b_sample = RGBSingleChannelDenoisingAESample(rgb_path, channel=0, mean= mean[3])
        self.seg_sample = ImageDenoisingAESample(s_path, is_grayscale=is_grayscale, mean= mean[4])

        super(FullMAESample, self).__init__()

    def read_features(self):

        d_f = self.d_sample.read_features()
        r_f = self.r_sample.read_features()
        g_f = self.g_sample.read_features()
        b_f = self.b_sample.read_features()
        s_f = self.seg_sample.read_features()

        #Sometimes feed the network with only rgb samples, setting depth and seg to 0
        if np.random.uniform() > 0.5:
            d_f = np.zeros_like(d_f)
            s_f = np.zeros_like(s_f)

        return [d_f, r_f, g_f, b_f, s_f]

    def read_labels(self):
        d_l = self.d_sample.read_labels()
        r_l = self.r_sample.read_labels()
        g_l = self.g_sample.read_labels()
        b_l = self.b_sample.read_labels()
        s_l = self.seg_sample.read_labels()

        return [d_l,r_l,g_l,b_l,s_l]

    def set_new_size(self, width, height):
        self.d_sample.set_new_size(width, height)
        self.r_sample.set_new_size(width, height)
        self.b_sample.set_new_size(width, height)
        self.g_sample.set_new_size(width, height)
        self.seg_sample.set_new_size(width, height)
