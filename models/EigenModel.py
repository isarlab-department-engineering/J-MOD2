import tensorflow as tf
from keras.models import Model
from keras.layers import Reshape, Flatten,Convolution2D, Lambda, Input, MaxPooling2D, Dropout, Cropping2D, ZeroPadding2D
from keras.layers import Dense
from keras.layers.merge import Concatenate
from keras.applications.vgg16 import VGG16
from lib.DepthObjectives import eigen_loss
from lib.DepthMetrics import rmse_metric, logrmse_metric, sc_inv_logrmse_metric


import numpy as np
import cv2

from DepthFCNModel import DepthFCNModel

from keras.optimizers import Adam, Adadelta, SGD


class EigenModel_Scale2(DepthFCNModel):
    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0
        label = np.asarray(label)
        label = label.astype('float32')
        # label = label * 39.75 / 255.0
        label = -4.586e-09 * (label ** 4) + 3.382e-06 * (label ** 3) - 0.000105 * (
        label ** 2) + 0.04239 * label + 0.04072
        label /= 39.75

        # label = tf.image.resize_images(label, [80, 128])

        label_resize = np.zeros(shape=(label.shape[0], 40, 64, 1))

        if label.shape[0] is not None:
            for i in range(0, label.shape[0]):
                label_resize[i, :, :, 0] = cv2.resize(label[i, :, :, 0], (64, 40), cv2.INTER_LINEAR)
            label = label_resize
        else:
            label = np.resize(label, (label.shape[0], 40, 64, 1))

        return features, label

    def define_architecture(self):
        input = Input(shape=(self.config.input_height,
                             self.config.input_width,
                             self.config.input_channel), name='input')

        # Scale 1
        vgg16model = VGG16(include_top=False, weights='imagenet', input_tensor=input,
                           input_shape=(self.config.input_height,
                                        self.config.input_width,
                                        self.config.input_channel
                                        ))

        # feature_map_1 = vgg16model.layers[6].output  # verificare 64x40
        scale_1 = vgg16model.layers[-1].output
        scale_1 = Lambda(lambda x: x * 0.01, name='vgg_scale')(scale_1)

        scale_1 = Flatten()(scale_1)

        scale_1 = Dense(4096, activation='relu')(scale_1)
        scale_1 = Dropout(0.5)(scale_1)
        scale_1 = Dense(10240, activation='relu')(scale_1)

        scale_1 = Reshape((10, 16, 64))(scale_1)

        scale_1 = Lambda(lambda x: tf.image.resize_images(x, [37, 61]), name='lambda_1')(scale_1)

        # Scale 2
        feature_map_1 = Convolution2D(96, (9, 9), activation='relu', padding='valid', strides=(2, 2),
                                      name='feature_map_1')(input)
        feature_map_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(feature_map_1)
        # feature_map_1 = Lambda(lambda x: tf.image.resize_images(x, [40, 64]), name='lambda_2')(feature_map_1)

        scale_2_input = Concatenate(axis=-1)([feature_map_1, scale_1])

        x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='scale2_conv1')(scale_2_input)
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale2_conv2')(x)
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale2_conv3')(x)
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale2_conv4')(x)
        scale2_out = Convolution2D(1, (5, 5), activation='relu', padding='same', name='out')(x)

        #if True:
        #    out = Lambda(lambda x: tf.image.resize_images(x, [160, 256]), name='test_out_up')(scale2_out)
        #else:
        out = Lambda(lambda x: tf.image.resize_images(x, [40, 64]), name='out_up')(scale2_out)
        model = Model(inputs=input, outputs=out)

        return model, input

    def build_model(self):

        fcn_model, input = self.define_architecture()

        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)
        fcn_model.compile(loss=eigen_loss,
                          optimizer=opt,
                          metrics=[rmse_metric, logrmse_metric])

        fcn_model.summary()
        return fcn_model


class EigenModel_Scale3(EigenModel_Scale2):
    def __init__(self, config, scale2_weights, upsample_to_original = False):
        self.crop_w_dim = int(np.random.uniform() * (128. - 64.))  # will be recomputed at each batch
        self.crop_h_dim = int(np.random.uniform() * (80. - 40.))
        self.upsample_to_original = upsample_to_original
        self.scale2_weights = scale2_weights
        super(EigenModel_Scale3, self).__init__(config)

    def prepare_data_for_model(self, features, label):
        features = np.asarray(features)
        features = features.astype('float32')

        features /= 255.0
        label = np.asarray(label)
        label = label.astype('float32')
        # label = label * 39.75 / 255.0
        label = -4.586e-09 * (label ** 4) + 3.382e-06 * (label ** 3) - 0.000105 * (
        label ** 2) + 0.04239 * label + 0.04072
        label /= 39.75

        label_resize = np.zeros(shape=(label.shape[0], 80, 128, 1))
        """
        if label.shape[0] is not None:
            for i in range(0, label.shape[0]):
                label_resize[i, :, :, 0] = cv2.resize(label[i, :, :, 0], (128, 80), cv2.INTER_LINEAR)
            label = label_resize
        else:
            label = np.resize(label, (label.shape[0], 80, 128, 1))
        """
        """
        if self.config.is_train:
            # Crop label
            self.crop_w_dim = int(
                np.random.uniform() * (128 - 64))  # this will be read to model to crop scale2 prediction
            self.crop_h_dim = int(np.random.uniform() * (80 - 40))

            label_crop = label[:, 2*self.crop_h_dim:2*(self.crop_h_dim + 40), 2*self.crop_w_dim:2*(self.crop_w_dim + 64), :]
            label = label_crop
        """
        if label.shape[0] is not None:
            for i in range(0, label.shape[0]):
                label_resize[i, :, :, 0] = cv2.resize(label[i, :, :, 0], (128, 80), cv2.INTER_LINEAR)
            label = label_resize
        else:
            label = np.resize(label, (label.shape[0], 80, 128, 1))

        return features, label

    def define_architecture(self):
        scale_2_model, input = EigenModel_Scale2(self.config).define_architecture()
        scale_2_model.load_weights(self.scale2_weights)

        for layer in scale_2_model.layers:
            if layer.name is not 'Lambda':
                layer.trainable = False

        scale2_out = scale_2_model(input)

        scale2_out = Lambda(lambda x: tf.image.resize_images(x, [80, 128]), name='out_scale3_up')(scale2_out)


            #input_crop = Cropping2D(cropping=((self.crop_h_dim , 160 - self.crop_h_dim - 80 - 11),
            #                                  (self.crop_w_dim , 256 - self.crop_w_dim - 128 - 11)))(input)
        input_pad = ZeroPadding2D((6,6))(input)
        feature_map_2 = Convolution2D(64, (9, 9), activation='relu', padding='valid', strides=(2, 2), name='feature_map_2')(input_pad)
        feature_map_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(feature_map_2)
        #feature_map_2 = Lambda(lambda x: tf.image.resize_images(x, [40, 64]), name='lambda_1')(feature_map_2)


        # Scale 3
        scale_3_input = Concatenate(axis=-1)([feature_map_2, scale2_out])
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale3_conv1')(scale_3_input)
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale3_conv2')(x)
        x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='scale3_conv3')(x)
        out = Convolution2D(1, (5, 5), activation='relu', padding='same', name='scale3_out')(x)
        """
        if self.config.is_train:
            out = Lambda(lambda x: tf.image.resize_images(x, [40, 64]), name='test_out_up')(out)
        else:
            out = Lambda(lambda x: tf.image.resize_images(x, [80, 128]), name='test_out_up')(out)
        """
        if self.upsample_to_original:
            out = Lambda(lambda x: tf.image.resize_images(x, [160, 256]), name='test_out_up')(out)
        eigen_model = Model(inputs=input, outputs=out)

        return eigen_model

    def build_model(self):
        eigen_model = self.define_architecture()

        opt = Adam(lr=self.config.learning_rate, clipnorm=1.)
        eigen_model.compile(loss=eigen_loss,
                            optimizer=opt,
                            metrics=[rmse_metric, logrmse_metric])

        eigen_model.summary()
        return eigen_model

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

        pred_depth = net_output * 39.75

        return [pred_depth,None,None]