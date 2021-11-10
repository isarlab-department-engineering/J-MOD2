import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
	return v.lower() in ('true', '1')

def add_argument_group(name):
	arg = parser.add_argument_group(name)
	arg_lists.append(arg)
	return arg

# Network
net_arg = add_argument_group('Network')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_set_dir', type=str, default='data/UnrealDataset')
data_arg.add_argument('--use_subsampled', type=str2bool, default=False)
data_arg.add_argument('--compute_mean', type=str2bool, default=False)
data_arg.add_argument('--data_main_dir', type=str, default='')
data_arg.add_argument('--data_train_dirs', type=eval, nargs='+', default=['00_D', '01_D', '02_D', '03_D', '04_D', '05_D', '06_D', '07_D', '08_D', '10_D', '11_D', '13_D', '15_D', '16_D', '17_D', '18_D', '19_D', '20_D'])
data_arg.add_argument('--data_test_dirs', type=eval, nargs='+', default=['09_D', '14_D'])
data_arg.add_argument('--input_height', type=int, default=160)
data_arg.add_argument('--input_width', type=int, default=256)
data_arg.add_argument('--input_channel', type=int, default=3)
data_arg.add_argument('--img_extension', type=str, default="png")

#JMOD2 param
jmod2_arg = add_argument_group('JMOD2')
jmod2_arg.add_argument('--detector_confidence_thr', type=int, default=0.5)
jmod2_arg.add_argument('--non-max_suppresion', type=str2bool, default=False)

#CADENA param
cad_arg = add_argument_group('Cadena')
cad_arg.add_argument('--cadena_resize_out', type=str2bool, default=True)
#Used for training only
cad_arg.add_argument('--rgb_ae_channel_override', type=str2bool, default=False)
cad_arg.add_argument('--rgb_ae_channel', type=int, default=2)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='') #Set True for training
train_arg.add_argument('--exp_name', type=str, default='NAME_OF_EXPERIMENT')
train_arg.add_argument('--preload_ram',type=str2bool, default=False)
train_arg.add_argument('--use_augmentation', type=str2bool, default=True, help='')
train_arg.add_argument('--validation_split', type=float, default=0.2, help='')
train_arg.add_argument('--max_step', type=int, default=10000, help='')
train_arg.add_argument('--batch_size', type=int, default=64, help='')
train_arg.add_argument('--buffer_size', type=int, default=25600, help='')
train_arg.add_argument('--num_epochs', type=int, default=100, help='')
train_arg.add_argument('--random_seed', type=int, default=123, help='')
train_arg.add_argument('--learning_rate', type=float, default=1e-4, help='')
train_arg.add_argument('--weights_path', type=str, default="weights/jmod2.hdf5") #Used for finetuning or to resume training
train_arg.add_argument('--weights_path_vgg19', type=str, default="weights/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
train_arg.add_argument('--resume_training', type=str2bool, default=False)
train_arg.add_argument('--version_model', type=int, default=1)#1:original version J-MOD2

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--is_deploy', type=str2bool, default=False, help='')
misc_arg.add_argument('--log_step', type=int, default=20, help='')
misc_arg.add_argument('--log_dir', type=str, default='logs') #DIRECTORY WHERE TO SAVE MODEL CHECKPOINTS
misc_arg.add_argument('--debug', type=str2bool, default=True)
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=0.8)
misc_arg.add_argument('--max_image_summary', type=int, default=4)

def get_config():
	config, unparsed = parser.parse_known_args()
	return config, unparsed