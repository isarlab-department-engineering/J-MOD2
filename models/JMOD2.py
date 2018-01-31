from keras.models import Model
from keras.layers import  Reshape, Flatten,Convolution2D,  Dropout, Concatenate
from keras.layers import Dense

from lib.SampleType import DepthObstacles_SingleFrame

from lib.DepthObjectives import root_mean_squared_logarithmic_loss, root_mean_squared_loss, mean_squared_loss, log_normals_loss,eigen_loss
from lib.ObstacleDetectionObjectives import yolo_v1_loss, iou_metric, recall, precision, mean_metric, variance_metric
from lib.DepthMetrics import rmse_metric, logrmse_metric, sc_inv_logrmse_metric

import numpy as np

from DepthFCNModel import DepthFCNModel

from lib.Dataset import UnrealDataset_DepthSupervised
from lib.DataGenerationStrategy import SingleFrameGenerationStrategy, PairGenerationStrategy

from keras.optimizers import Adam, Adadelta, SGD

from lib.EvaluationUtils import get_detected_obstacles_from_detector

class JMOD2(DepthFCNModel):

    def load_dataset(self):
        dataset = UnrealDataset_DepthSupervised(self.config, SingleFrameGenerationStrategy(sample_type=DepthObstacles_SingleFrame,
                                                                           get_obstacles=True), read_obstacles=True)
        dataset.data_generation_strategy.mean = dataset.mean
        dataset_name = 'UnrealDataset'

        return dataset, dataset_name

    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0

        labels_depth = np.zeros(shape=(features.shape[0],features.shape[1],features.shape[2],1), dtype=np.float32)
        labels_obs = np.zeros(shape=(features.shape[0],40,7), dtype=np.float32)
        i = 0
        for elem in label:
            elem["depth"] = np.asarray(elem["depth"]).astype(np.float32)

            elem["depth"] = -4.586e-09 * (elem["depth"] ** 4) + 3.382e-06 * (elem["depth"] ** 3) - 0.000105 * (elem["depth"] ** 2) + 0.04239 * elem["depth"] + 0.04072
            elem["depth"] /= 39.75

            labels_depth[i,:,:,:] = elem["depth"]
            labels_obs[i,:,:] = np.asarray(elem["obstacles"]).astype(np.float32)
            i +=1

        return features, [labels_depth,labels_obs]

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


        model = Model(inputs= depth_model.inputs[0], outputs=[depth_model.outputs[0], out_detection])

        opt = Adam(lr=self.config.learning_rate, clipnorm = 1.)
        model.compile(loss={'depth_output': log_normals_loss, 'detection_output':yolo_v1_loss},
                            optimizer=opt,
                            metrics={'depth_output': [rmse_metric, logrmse_metric, sc_inv_logrmse_metric], 'detection_output': [iou_metric, recall, precision, mean_metric, variance_metric]},
                            loss_weights=[1.0, 1.0])
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

        #print ("Elapsed time: {}").format(time.time() - t0)

        pred_depth = net_output[0] * 39.75
        pred_detection = net_output[1]

        pred_obstacles, rgb_with_detection = get_detected_obstacles_from_detector(pred_detection, self.config.detector_confidence_thr)

        correction_factor = self.compute_correction_factor(pred_depth, pred_obstacles)

        corrected_depth = np.array(pred_depth) * correction_factor

        return [pred_depth,pred_obstacles,corrected_depth]

    def compute_correction_factor(self, depth, obstacles):

        mean_corr = 0
        it = 0

        for obstacle in obstacles:
            depth_roi = depth[0, np.max((obstacle.y, 0)):np.min((obstacle.y + obstacle.h, depth.shape[1] - 1)),
                        np.max((obstacle.x, 0)):np.min((obstacle.x + obstacle.w, depth.shape[2] - 1)), 0]

            if len(depth_roi) > 0:
                mean_est = np.mean(depth_roi)
                it += 1
                mean_corr += obstacle.depth_mean / mean_est

        if it > 0:
            mean_corr /= it
        else:
            mean_corr = 1
        return mean_corr





