import os
from glob import glob
import time
import sys

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import numpy as np
import math
import matplotlib.pyplot as plt
from lib.DepthCallback import PrintBatch, TensorBoardCustom
from lib.DepthMetrics import rmse_error, logrmse_error, normals_metric
import tensorflow as tf

class AbstractModel(object):

    def __init__(self, config, shuffle_data = True):
        self.config = config
        self.dataset = {}
        self.training_set = {}
        self.test_set = {}
        self.validation_set = {}
        self.input_w = self.config.input_width
        self.input_h = self.config.input_height
        self.data_augmentation_strategy = None
        self.model = self.build_model()
        self.mean = None

        self.prepare_data()
        self.shuffle = shuffle_data


    def prepare_data(self):

        raise ("Not implemented - this is an abstract method")


    def train_data_generator(self):

        if self.shuffle:
            np.random.shuffle(self.training_set)
        curr_batch = 0
        self.training_set = list(self.training_set)
        while 1:

            if (curr_batch + 1) * self.config.batch_size > len(self.training_set):
                np.random.shuffle(self.training_set)
                curr_batch = 0
            t0 = time.time()

            x_train = []
            y_train = []

            for sample in self.training_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:

                features = sample.read_features()
                label = sample.read_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=False)

                x_train.append(features)
                y_train.append(label)

            t1 = time.time()

            #print("Loading completed in " + str(t1 - t0) + " seconds")

            x_train, y_train = self.prepare_data_for_model(x_train, y_train)

            curr_batch += 1

            yield x_train , y_train

    def validation_data_generator(self):

        if self.shuffle:
            np.random.shuffle(self.training_set)
        curr_batch = 0

        while 1:

            if (curr_batch + 1) * self.config.batch_size > len(self.validation_set):
                np.random.shuffle(self.validation_set)
                curr_batch = 0

            x_train = []
            y_train = []

            for sample in self.validation_set[curr_batch * self.config.batch_size: (curr_batch + 1) * self.config.batch_size]:
                features = sample.read_features()
                label = sample.read_labels()
                if self.data_augmentation_strategy is not None:
                    features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)

                x_train.append(features)
                y_train.append(label)


            x_train, y_train = self.prepare_data_for_model(x_train, y_train)
            curr_batch += 1
            yield x_train , y_train

    def test_data_generator(self):

        curr_test_sample = 0

        while 1:
            if (curr_test_sample == len(self.test_set)):
                break

            x_test = self.test_set[curr_test_sample].read_features()
            y_test = self.test_set[curr_test_sample].read_labels()
            print(np.shape(y_test))

            x_test, y_test = self.prepare_data_for_model(x_test, y_test)
            x_test = np.expand_dims(x_test, axis=0)
            curr_test_sample += 1

            yield x_test , y_test

    def tensorboard_data_generator(self, num_samples):

        curr_train_sample_list = self.training_set[0:num_samples]
        curr_validation_sample_list = self.validation_set[0: num_samples]
        x_train = []
        y_train = []

        for sample in curr_train_sample_list:
            features = sample.read_features()
            label = sample.read_labels()
            if self.data_augmentation_strategy is not None:
                features, label = self.data_augmentation_strategy.process_sample(features, label)

            x_train.append(features)
            y_train.append(label)

        for sample in curr_validation_sample_list:
            features = sample.read_features()
            label = sample.read_labels()
            if self.data_augmentation_strategy is not None:
                features, label = self.data_augmentation_strategy.process_sample(features, label, is_test=True)

            x_train.append(features)
            y_train.append(label)

        x_train, y_train = self.prepare_data_for_model(x_train, y_train)
        return x_train , y_train

    def prepare_data_for_model(self, features, label):
        raise ("Not implemented - this is an abstract method")

    def define_architecture(self):
        raise ("Not implemented - this is an abstract method")

    def build_model(self):

        raise ("Not implemented - this is an abstract method")

    def train(self, initial_epoch=0):

        orig_stdout = sys.stdout
        f = open(os.path.join(self.config.model_dir, 'model_summary.txt'), 'w')
        sys.stdout = f

        print(self.model.summary())
        for layer in self.model.layers:
            print(layer.get_config())
        sys.stdout = orig_stdout
        f.close()

        plot_model(self.model, show_shapes=True, to_file=os.path.join(self.config.model_dir, 'model_structure.png'))

        t0 = time.time()
        samples_per_epoch = int(math.floor(len(self.training_set) / self.config.batch_size))
        val_step = int(math.floor(len(self.validation_set) / self.config.batch_size))
        print("Samples per epoch: {}".format(samples_per_epoch))
        pb = PrintBatch()
        tb_x, tb_y = self.tensorboard_data_generator(self.config.max_image_summary)
        tb = TensorBoardCustom(self.config, tb_x, tb_y, self.config.tensorboard_dir)
        model_checkpoint = ModelCheckpoint(
            os.path.join(self.config.model_dir, 'weights-{epoch:02d}-{loss:.2f}.hdf5'),
            monitor='loss', verbose=2,
            save_best_only=False, save_weights_only=False, mode='auto', period=self.config.log_step)
        self.model.fit_generator(generator=self.train_data_generator(),
                                 steps_per_epoch=samples_per_epoch,
                                 callbacks=[pb, model_checkpoint, tb],
                                 validation_data=self.validation_data_generator(),
                                 validation_steps=val_step,
                                 epochs=self.config.num_epochs,
                                 verbose=2,
                                 initial_epoch=initial_epoch)
        t1 = time.time()

        print("Training completed in " + str(t1 - t0) + " seconds")

    def resume_training(self, weights_file, initial_epoch):
        self.model.load_weights(weights_file)
        self.train(initial_epoch)

    def test(self, weights_file, showFigure=False):
        print("Testing model")
        self.model.load_weights(weights_file)
        acc_rmse_error = 0
        acc_logrmse_error = 0
        acc_normal_error_x = 0
        acc_normal_error_y = 0
        acc_frequency = 0
        curr_img = 0
        if showFigure:
            plt.figure(1)
            plt.title('Predicted Depth')
            plt.figure(2)
            plt.title('GT Depth')
            plt.figure(3)
            plt.title('Input')
        for x_test, y_test in self.test_data_generator():
            print('Testing img: ', curr_img)
            t0 = time.time()
            output = self.model.predict(x_test) * 39.75
            y_test = y_test * 39.75
            t1 = time.time()
            acc_frequency += 1 / (t1 - t0)
            print("Evaluation complete in " + str(t1 - t0) + " seconds")
            print("Evaluation Frequency " + str(1 / (t1 - t0)) + " hz")
            if showFigure:
                plt.figure(1)
                plt.clf()
                plt.imshow(output[0, :, :, 0])
                # imwrite('pred.png', output[0, :, :, 0])
                plt.figure(2)
                plt.clf()
                plt.imshow(y_test[:, :, 0])
                plt.figure(3)
                plt.imshow(x_test[0, :, :, :])
                # imwrite('gt.png', y_test[0, :, :, 0])
                plt.pause(0.05)

            acc_rmse_error += rmse_error(y_test, output)
            acc_logrmse_error += logrmse_error(y_test, output)
            #n_x, n_y = normals_metric(y_test, output)
            #acc_normal_error_x += n_x
            #acc_normal_error_y += n_y
            print('Rmse: ', rmse_error(y_test, output))
            print('Log RMSE: ', logrmse_error(y_test, output))
            #print('Normal rec error X axis: ', n_x)
            #print('Normal rec error Y axis: ', n_y)
            curr_img += 1

        print('Average rmse error: ')
        print(acc_rmse_error / len(self.test_set))

        print('Average log rmse error: ')
        print(acc_logrmse_error / len(self.test_set))

        #print('Average normals X axis reconstruction error: ')
        #print(acc_normal_error_x / len(self.test_set))

        #print('Average normals Y axis reconstruction error: ')
        #print(acc_normal_error_y / len(self.test_set))

        print('Average frequency: ')
        print(acc_frequency / len(self.test_set))