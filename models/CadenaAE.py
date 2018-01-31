import keras.optimizers

from lib.DepthObjectives import mean_squared_loss
import time

import cv2
import keras.optimizers
import matplotlib.pyplot as plt
from lib.DataGenerationStrategy import *
from lib.Dataset import UnrealDataset_Autoencoder
from keras.layers import Input, Dense, Flatten, Reshape, Concatenate
from keras.models import Model

from AbstractModel import AbstractModel
from lib.SegmentationMetrics import SegmentationMetrics

import lib.EvaluationUtils as EvaluationUtils

from lib.DataAugmentationStrategy import HorFlipAugmentationStrategy

class BaseDenoisingAE(AbstractModel):
    def __init__(self, config, augmentation = True, load_samples = True):
        self.is_deploy = config.is_deploy

        self.use_augmentation = augmentation

        if not load_samples:
            self.is_deploy = True

        super(BaseDenoisingAE, self).__init__(config)


    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_Autoencoder(self.config, 'segmentation', SampleGenerationStrategy)
            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'UnrealDataset'
            return dataset, dataset_name
    def prepare_data(self):

        if not self.is_deploy:
            dataset, dataset_name = self.load_dataset()
            dataset.data_generation_strategy.config = self.config

            if self.use_augmentation:
                self.data_augmentation_strategy = HorFlipAugmentationStrategy(0.5)

            self.dataset[dataset_name] = dataset
            self.dataset[dataset_name].read_data()

            if dataset_name is not "Unreal_FULL":
                self.dataset[dataset_name].compute_stats()
            #self.dataset[dataset_name].print_info(show_sequence=False)

            #self.data_augmentation_strategy = DepthDataAugmentationStrategy()

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
    def prepare_data_for_model(self, features, label):

        features = np.asarray(features)
        features = features.astype('float32')
        #features *= 255.0
        label = np.asarray(label)
        label = label.astype('float32')

        return features, label

    def define_architecture(self):
        input = Input(shape=(self.input_h,
                             self.input_w,
                             1), name='input')
        input_ = Flatten()(input)

        hidden = Dense(1024,activation='relu',kernel_initializer='random_uniform',name='single_mode_hidden')(input_)
        output = Dense(self.input_w*self.input_h,kernel_initializer='random_uniform',activation='sigmoid')(hidden)
        output_im = Reshape((self.input_h,self.input_w,1))(output)

        model = Model(inputs=input, outputs=output_im)

        return model
    def build_model(self):

        model = self.define_architecture()
        #adadelta = keras.optimizers.Adadelta()
        sgd = keras.optimizers.SGD(lr=self.config.learning_rate)
        model.compile(loss=mean_squared_loss,
                    optimizer=sgd)

        model.summary()
        return model

class SegmentationDenoisingAE(BaseDenoisingAE):
    def __init__(self, config, augmentation = True, load_samples = True):
        self.is_deploy = config.is_deploy

        super(SegmentationDenoisingAE, self).__init__(config, augmentation, load_samples)

    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_Autoencoder(self.config, 'segmentation2', SegmentationAutoEncoderGenerationStrategy, 'Unreal_Segmentation')
            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'unreal_segm'
            return dataset, dataset_name
    def define_architecture(self):
        input = Input(shape=(self.input_h,
                             self.input_w,
                             1), name='seg_input')
        input_ = Flatten()(input)

        hidden = Dense(1024,activation='relu',kernel_initializer='random_uniform',name='seg_single_mode_hidden')(input_)
        output = Dense(self.input_w*self.input_h,kernel_initializer='random_uniform',activation='sigmoid')(hidden)
        output_im = Reshape((self.input_h,self.input_w,1))(output)

        model = Model(inputs=input, outputs=output_im)

        return model

    def test(self, weights_file, showFigure=False):
        print("Testing model")
        self.model.load_weights(weights_file)
        curr_img = 0
        if showFigure:
            plt.figure(1)
            plt.title('Reconstruction')
            plt.figure(2)
            plt.title('GT')
        acc_frequency = 0
        metrics = SegmentationMetrics()
        for x_test, y_test in self.test_data_generator():
            print('Testing img: ', curr_img)
            t0 = time.time()
            output = self.model.predict(x_test)
            y_test = y_test
            t1 = time.time()
            acc_frequency += 1 / (t1 - t0)
            thr_out = cv2.threshold(output[0, :, :, 0],0.5,1,cv2.THRESH_BINARY)[1]
            metrics.update(thr_out,y_test[:, :, 0] )

            print("Evaluation complete in " + str(t1 - t0) + " seconds")
            print("Evaluation Frequency " + str(1 / (t1 - t0)) + " hz")
            pa,ma,iou,wiou = metrics.get_stats()
            print(" Pixel accuracy: {}\n Mean accuracy: {} \n Mean IOU: {}\n Frequency Weighted IOU: {}\n").format(pa,ma,iou,wiou)
            if showFigure:
                plt.figure(1)
                plt.clf()
                plt.imshow(thr_out)
                # imwrite('pred.png', output[0, :, :, 0])
                plt.figure(2)
                plt.clf()
                plt.imshow(x_test[0, :, :, 0])
                # imwrite('gt.png', y_test[0, :, :, 0])
                plt.pause(0.005)

            curr_img += 1

        print('Average frequency: ')
        print(acc_frequency / len(self.test_set))


class RGBDenoisingAE(BaseDenoisingAE):
    def __init__(self, config, channel = 0, augmentation = True, load_samples = True):
        #self.config = config
        self.is_deploy = config.is_deploy
        self.channel = channel
        super(RGBDenoisingAE, self).__init__(config, augmentation, load_samples)

        self.config.rgb_ae_channel = channel

    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_Autoencoder(self.config, 'rgb', RGB_SingleChannel_AutoEncoderGenerationStrategy, 'Unreal_RGB')
            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'unreal_rgb'
            return dataset, dataset_name
    def test(self, weights_file, showFigure=False):
        print("Testing model")
        self.model.load_weights(weights_file)
        curr_img = 0
        if showFigure:
            plt.figure(1)
            plt.title('Reconstruction')
            plt.figure(2)
            plt.title('GT')
        acc_frequency = 0
        for x_test, y_test in self.test_data_generator():
            print('Testing img: ', curr_img)
            t0 = time.time()
            output = self.model.predict(x_test)
            y_test = y_test
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
                # imwrite('gt.png', y_test[0, :, :, 0])
                plt.pause(0.005)

            curr_img += 1

        print('Average frequency: ')
        print(acc_frequency / len(self.test_set))

    def define_architecture(self):

        name_pref = 'channel_{}'.format(self.channel)

        input = Input(shape=(self.input_h,
                             self.input_w,
                             1), name=name_pref + '_input')
        input_ = Flatten()(input)

        hidden = Dense(1024,activation='relu',kernel_initializer='random_uniform',name=name_pref + '_single_mode_hidden')(input_)
        output = Dense(self.input_w*self.input_h,kernel_initializer='random_uniform',activation='sigmoid')(hidden)
        output_im = Reshape((self.input_h,self.input_w,1))(output)

        model = Model(inputs=input, outputs=output_im)

        return model

class DepthDenoisingAE(BaseDenoisingAE):
    def __init__(self, config, augmentation = True, load_samples = True):
        self.config = config
        self.is_deploy = config.is_deploy

        super(DepthDenoisingAE, self).__init__(config, augmentation, load_samples)

    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_Autoencoder(self.config, 'depth', DepthAutoEncoderGenerationStrategy, 'Unreal_Depth')
            dataset.data_generation_strategy.mean = dataset.mean
            dataset_name = 'unreal_depth'
            return dataset, dataset_name
    def define_architecture(self):
        input = Input(shape=(self.input_h,
                             self.input_w,
                             1), name='depth_input')
        input_ = Flatten()(input)

        hidden = Dense(1024,activation='relu',kernel_initializer='random_uniform',name='depth_single_mode_hidden')(input_)
        output = Dense(self.input_w*self.input_h,kernel_initializer='random_uniform',activation='linear')(hidden)
        output_im = Reshape((self.input_h, self.input_w, 1))(output)

        model = Model(inputs=input, outputs=output_im)

        return model

class FullMAE(BaseDenoisingAE):

    def __init__(self,config, init_phase = True):
        self.depth_ae = DepthDenoisingAE(config, load_samples=False)
        self.r_ae = RGBDenoisingAE(config,channel=2, load_samples=False)
        self.g_ae = RGBDenoisingAE(config,channel=1, load_samples=False)
        self.b_ae = RGBDenoisingAE(config,channel=0, load_samples=False)
        self.seg_ae = SegmentationDenoisingAE(config, load_samples=False)
        self.config = config
        self.is_deploy = config.is_deploy
        self.init_phase = init_phase

        self.depth_model = self.depth_ae.define_architecture()
        self.r_model = self.r_ae.define_architecture()
        self.b_model = self.b_ae.define_architecture()
        self.g_model = self.g_ae.define_architecture()
        self.seg_model = self.seg_ae.define_architecture()

        if init_phase:
            #train from single task denoising AE to full MAE at smallest res
            for layer in self.depth_model.layers:
                if layer.name is 'depth_single_mode_hidden':
                    layer.trainable = False
            for layer in self.r_model.layers:
                if 'hidden' in layer.name:
                    layer.trainable = False
            for layer in self.g_model.layers:
                if 'hidden' in layer.name:
                    layer.trainable = False
            for layer in self.b_model.layers:
                if 'hidden' in layer.name:
                    layer.trainable = False
            for layer in self.seg_model.layers:
                if layer.name is 'seg_single_mode_hidden':
                    layer.trainable = False

        super(FullMAE, self).__init__(config)

        if self.config.is_train:
            self.load_weights()



    def load_dataset(self):
        if self.config.dataset is 'UnrealDataset':
            dataset = UnrealDataset_Autoencoder(self.config, 'full', FullMAEGenerationStrategy, 'Unreal_FULL')

            dataset.data_generation_strategy.mean = [np.load('{}/{}/{}_mean.npy'.format(self.config.data_set_dir, self.config.data_main_dir, "Unreal_Depth")),
                                                     np.load('{}/{}/{}_mean.npy'.format(self.config.data_set_dir,
                                                                                        self.config.data_main_dir,
                                                                                        "Unreal_RGB")),
                                                     np.load('{}/{}/{}_mean.npy'.format(self.config.data_set_dir,
                                                                                        self.config.data_main_dir,
                                                                                        "Unreal_RGB")),
                                                     np.load('{}/{}/{}_mean.npy'.format(self.config.data_set_dir,
                                                                                        self.config.data_main_dir,
                                                                                        "Unreal_RGB")),
                                                     np.load('{}/{}/{}_mean.npy'.format(self.config.data_set_dir,
                                                                                        self.config.data_main_dir,
                                                                                            "Unreal_Segmentation")),]
            dataset_name = 'Unreal_FULL'
            return dataset, dataset_name

    def load_weights(self):
        # train from smallest res to bigger res
        if True:

            weights_file = "/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/FULL_AE_DATACORRUPTED_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-30_17-19-24/weights-149-0.39.hdf5"
            import h5py
            weights = h5py.File(weights_file,'r')
            shared_in_sem_hidden_weights = weights['model_weights']['shared_in_sem_hidden']['shared_in_sem_hidden'] #keys: bias:0 kernel:0
            shared_hidden_weights = weights['model_weights']['shared_hidden']['shared_hidden'] #keys: bias:0 kernel:0
            shared_sem_out_weights = weights['model_weights']['shared_out_sem_hidden']['shared_out_sem_hidden']

            self.model.layers[15].set_weights([shared_in_sem_hidden_weights['kernel:0'],shared_in_sem_hidden_weights['bias:0']])
            self.model.layers[17].set_weights([shared_hidden_weights['kernel:0'],shared_hidden_weights['bias:0']])
            self.model.layers[18].set_weights([shared_sem_out_weights['kernel:0'], shared_sem_out_weights['bias:0']])

        elif self.init_phase:
            self.depth_model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/DEPTH_AE_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-28_16-46-15/weights-149-0.02.hdf5")
            self.r_model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/R_AE_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-29_17-40-14/weights-149-0.01.hdf5")
            self.g_model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/G_AE_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-29_14-46-02/weights-149-0.01.hdf5")
            self.b_model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/B_AE_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-29_11-17-43/weights-149-0.01.hdf5")
            self.seg_model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/SEG_AE_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-28_13-47-52/weights-149-0.08.hdf5")
        else:
            self.model.load_weights("/media/isarlab/e1a54258-46b5-46ed-81f7-5cf01e504c48/CadenaMultimodalAE/logs/FULL_AE_DATACORRUPTED_MSE_UnrealDataset_40_64_test_dirs_['09_D', '14_D']_2017-11-30_17-19-24/weights-149-0.39.hdf5")
    def prepare_data_for_model(self, features, label):

        list_feat = [np.asarray(features[:])[:,0,:,:,:].astype('float32'),
                     np.asarray(features[:])[:,1, :, :, :].astype('float32'),
                     np.asarray(features[:])[:,2, :, :, :].astype('float32'),
                     np.asarray(features[:])[:,3, :, :, :].astype('float32'),
                     np.asarray(features[:])[:,4, :, :, :].astype('float32'),
                     ]

        list_label = [np.asarray(label[:])[:,0,:,:,:].astype('float32'),
                     np.asarray(label[:])[:,1, :, :, :].astype('float32'),
                     np.asarray(label[:])[:,2, :, :, :].astype('float32'),
                     np.asarray(label[:])[:,3, :, :, :].astype('float32'),
                     np.asarray(label[:])[:,4, :, :, :].astype('float32'),
                      ]

        #cv2.imshow("depth", list_feat[0][0,...])
        #cv2.imshow("r", list_feat[1][0,...])
        #cv2.imshow("seg", list_feat[4][0,...])
        #cv2.waitKey(50)

        return list_feat, list_label



    def define_architecture(self):

        depth_feat = self.depth_model.layers[-3].output
        seg_feat = self.seg_model.layers[-3].output
        r_feat = self.r_model.layers[-3].output
        g_feat = self.g_model.layers[-3].output
        b_feat = self.b_model.layers[-3].output

        concat_feat = Concatenate()([depth_feat,r_feat,g_feat,b_feat,seg_feat])

        shared_feat = Dense(1024,activation='relu',kernel_initializer='random_normal',name='shared_hidden')(concat_feat)

        d_output = Dense(self.input_w * self.input_h, kernel_initializer='random_normal',activation='relu')(shared_feat)
        r_output = Dense(self.input_w * self.input_h, kernel_initializer='random_normal',activation='relu')(shared_feat)
        g_output = Dense(self.input_w * self.input_h, kernel_initializer='random_normal',activation='relu')(shared_feat)
        b_output = Dense(self.input_w * self.input_h, kernel_initializer='random_normal',activation='relu')(shared_feat)
        s_output = Dense(self.input_w * self.input_h, kernel_initializer='random_normal',activation='relu')(shared_feat)

        d_output_im = Reshape((self.input_h, self.input_w, 1),name='depth')(d_output)
        r_output_im = Reshape((self.input_h, self.input_w, 1),name='r')(r_output)
        g_output_im = Reshape((self.input_h, self.input_w, 1),name='g')(g_output)
        b_output_im = Reshape((self.input_h, self.input_w, 1),name='b')(b_output)
        s_output_im = Reshape((self.input_h, self.input_w, 1),name='seg')(s_output)


        model = Model(inputs=[self.depth_model.inputs[0],
                              self.r_model.inputs[0],
                              self.g_model.inputs[0],
                              self.b_model.inputs[0],
                              self.seg_model.inputs[0]],
                      outputs=[d_output_im,r_output_im,g_output_im,b_output_im,s_output_im])
        return model

    def run(self, input, resize=False):


        if len(input.shape) == 4:
            img = cv2.resize(input[0, :, :, :], (self.config.input_width, self.config.input_height), cv2.INTER_LINEAR)
            input = np.zeros(shape=(1, img.shape[0], img.shape[1], img.shape[2]))
            input[0, :, :, :] = img
            r_chan = input[:, :, :, 2]
            g_chan = input[:, :, :, 1]
            b_chan = input[:, :, :, 0]
        if len(input.shape) == 3:
            img = cv2.resize(input, (self.config.input_width, self.config.input_height), cv2.INTER_LINEAR)
            input = np.zeros(shape=(1, img.shape[0], img.shape[1], img.shape[2]))
            input[0, :, :, :] = img
            r_chan = input[:, :, :, 2]
            g_chan = input[:, :, :, 1]
            b_chan = input[:, :, :, 0]
        elif len(input.shape) == 2:
            input = cv2.resize(input, (self.config.input_width, self.config.input_height), cv2.INTER_LINEAR)
            input = np.expand_dims(input, 0)
            r_chan = input
            g_chan = input
            b_chan = input
        # caso 3

        depth = np.zeros_like(r_chan)
        seg = np.zeros_like(r_chan)

        r_chan = np.expand_dims(r_chan, axis=3).astype(np.float32)
        g_chan = np.expand_dims(g_chan, axis=3).astype(np.float32)
        b_chan = np.expand_dims(b_chan, axis=3).astype(np.float32)
        depth = np.expand_dims(depth, axis=3).astype(np.float32)
        seg = np.expand_dims(seg, axis=3).astype(np.float32)


        net_input = [depth, r_chan, g_chan, b_chan, seg]
        t0 = time.time()

        net_output = self.model.predict(net_input, batch_size=1)  # [0]

        pred_depth = net_output[0] * 39.75
        segm = net_output[4]


        if self.config.cadena_resize_out:
            resized_depth = np.zeros(shape=(1,160,256,1))
            resized_segm = np.zeros(shape=(1, 160, 256, 1))
            resized_depth[0,:,:,0] = cv2.resize(pred_depth[0,:,:,0], (256, 160), cv2.INTER_NEAREST)


            resized_segm[0,:,:,0] = cv2.resize(segm[0,:,:,0], (256, 160), cv2.INTER_NEAREST)

            pred_depth = resized_depth
            segm = resized_segm

        pred_obs = EvaluationUtils.get_obstacles_from_seg_and_depth(pred_depth, segm, segm_thr=-1, is_gt=True)

        #correction_factor = self.compute_correction_factor(pred_depth, pred_obs)
        #corrected_depth = np.array(pred_depth) * correction_factor

        print ("Elapsed time: {}").format(time.time() - t0)

        return [pred_depth, pred_obs, None]

    def compute_correction_factor(self, depth, detection):
        # convert detection to obstacles

        obstacles = detection

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