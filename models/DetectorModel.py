from keras.models import Model
from keras.layers import  Reshape, Flatten,Convolution2D,  Dropout, Concatenate

from lib.ObstacleDetectionObjectives import yolo_v1_loss, iou_metric, recall, precision, mean_metric, variance_metric

import numpy as np

from JMOD2 import JMOD2
from lib.EvaluationUtils import get_detected_obstacles_from_detector

from keras.optimizers import Adam, Adadelta, SGD

class Detector(JMOD2):
    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0
        labels_obs = np.zeros(shape=(features.shape[0],40,7), dtype=np.float32)
        i = 0
        for elem in label:
            labels_obs[i,:,:] = np.asarray(elem["obstacles"]).astype(np.float32)
            i +=1

        return features, labels_obs

    def build_model(self):
        depth_model = self.define_architecture()
        #Detection section
        output = depth_model.layers[-10].output

        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv1')(output)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv2')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv3')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv4')(x)
        x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='det_conv5')(x)

        x = Convolution2D(280, (3, 3), activation='relu', padding='same', name='det_conv6')(x)
        x = Reshape((40, 7, 160))(x)

        x = Convolution2D(160, (3, 3), activation='relu', padding='same', name='det_conv7')(x)
        x = Convolution2D(40, (3, 3), activation='relu', padding='same', name='det_conv8')(x)
        x = Convolution2D(1, (3, 3), activation='linear', padding='same', name='det_conv9')(x)

        out_detection = Reshape((40, 7), name='detection_output')(x)


        model = Model(inputs= depth_model.inputs[0], output= out_detection)

        #depth_model.load_weights("/home/isarlab/src/deepdepthestimation4oa/DepthEstimation/logs/normals_augmented_normals_UnrealDataset_160_256_test_dirs_['09_D', '14_D']_2017-08-18_19-46-55/weights-59-0.05.hdf5")

        opt = Adam(lr=self.config.learning_rate, clipnorm = 1.)
        model.compile(loss={'detection_output':yolo_v1_loss},
                            optimizer=opt,
                            metrics={'detection_output': [iou_metric, recall, precision, mean_metric, variance_metric]}) #1.0 1.0 per v1
        model.summary()
        return model

    def run(self, input):
        #import time

        mean = np.load('Unreal_RGB_mean.npy')

        if len(input.shape) == 2 or input.shape[2] == 1:
            tmp = np.zeros(shape=(input.shape[0],input.shape[1],3))
            tmp[:,:,0] = input
            tmp[:,:,1] = input
            tmp[:,:,2] = input

            input = tmp

        if len(input.shape) == 3:
            input = np.expand_dims(input-mean/255., 0)
        else:
            input[0,:,:,:] -= mean/255.

        #t0 = time.time()

        net_output = self.model.predict(input)

        pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector(net_output[0],
                                                                                  self.config.detector_confidence_thr)

        #print ("Elapsed time: {}").format(time.time() - t0)

        return [None,pred_obstacles,None]