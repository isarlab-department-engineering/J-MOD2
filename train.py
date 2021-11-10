import sys
import numpy as np
import tensorflow as tf

from models.JMOD2 import JMOD2
from lib.trainer import Trainer
from config import get_config
from lib.utils import prepare_dirs

config = None

def main(_):
	# Log dirs
	prepare_dirs(config)
	# Model JMOD2
	rng = np.random.RandomState(config.random_seed)
	tf.set_random_seed(config.random_seed)
	model = JMOD2(config)
	# Prepare trainer
	trainer = Trainer(config, model, rng)
	# train
	if config.is_train:
		trainer.train()
		#trainer.resume_training()
	else:
		trainer.test(showFigure=True)

if __name__ == "__main__":
	config, unparsed = get_config()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)