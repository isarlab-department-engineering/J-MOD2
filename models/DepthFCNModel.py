import tensorflow as tf
from keras.models import Model
from keras.layers import Convolution2D, Input, Conv2DTranspose
from keras.applications.vgg19 import VGG19

from keras.layers.advanced_activations import PReLU,LeakyReLU

from lib.DepthObjectives import root_mean_squared_logarithmic_loss, root_mean_squared_loss, mean_squared_loss, log_normals_loss,eigen_loss
from lib.DepthMetrics import rmse_metric, logrmse_metric, sc_inv_logrmse_metric
from lib.DataGenerationStrategy import SingleFrameGenerationStrategy
from lib.SampleType import Depth_SingleFrame
from lib.Dataset import UnrealDataset_DepthSupervised

import numpy as np
from models.FullyConvolutionalModel import FullyConvolutionalModel

from keras.optimizers import Adam, Adadelta, SGD


class DepthFCNModel(FullyConvolutionalModel):

    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_DepthSupervised(self.config, SingleFrameGenerationStrategy(sample_type=Depth_SingleFrame))
            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'UnrealDataset'
            return dataset, dataset_name


    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0

        label = np.asarray(label)
        label = label.astype('float32')
        #label = label * 39.75 / 255.0
        label = -4.586e-09 * (label ** 4) + 3.382e-06 * (label ** 3) - 0.000105 * (label ** 2) + 0.04239 * label + 0.04072
        label /= 39.75


        return features, label

    def define_architecture(self):
        input = Input(shape=(self.config.input_height,
                             self.config.input_width,
                             self.config.input_channel), name='input')

        vgg19model = VGG19(include_top=False, weights='imagenet', input_tensor=input,
                           input_shape=(self.config.input_height,
                                        self.config.input_width,
                                        self.config.input_channel
                                        ))

        # for layer in vgg19model.layers[0:12]:
        #    layer.trainable = False

        vgg19model.layers.pop()

        output = vgg19model.layers[-1].output

        # output = Dropout(0.15)(output)

        x = Conv2DTranspose(128, (4, 4), padding="same", strides=(2, 2))(output)
        x = PReLU()(x)
        x = Conv2DTranspose(64, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        # x = GaussianDropout(0.2)(x)
        x = Conv2DTranspose(32, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        x = Conv2DTranspose(16, (4, 4), padding="same", strides=(2, 2))(x)
        x = PReLU()(x)
        out = Convolution2D(1, (5, 5), padding="same", activation="relu", name="depth_output")(x)

        # out = Lambda(lambda x: K.clip(x,0.0,40.0))(out)

        # x = Convolution2D(1, 5, 5, border_mode='same', activation='relu', name='out')(x)
        ################### VGG Section - END ######################

        ################### DECONV Section - BEGIN ######################

        model = Model(inputs=input, outputs=out)

        return model

    def build_model(self):

        fcn_model = self.define_architecture()

        opt = Adam()
        fcn_model.compile(loss=log_normals_loss,
                            optimizer=opt,
                            metrics=[rmse_metric, logrmse_metric])

        fcn_model.summary()
        return fcn_model

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

        pred_depth = np.expand_dims(pred_depth,0)

        return [pred_depth,None,None]

