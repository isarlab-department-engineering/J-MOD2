import sys

import numpy as np
import tensorflow as tf
from models.JMOD2 import JMOD2
from models.DetectorModel import Detector
#import whatever model you need to train here
from lib.trainer import Trainer

from config import get_config
from lib.utils import prepare_dirs

config = None

def main(_):

  prepare_dirs(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  model = JMOD2(config)
  trainer = Trainer(config, model, rng)

  if config.is_train:
     trainer.train()
     #trainer.resume_training()
  else:
     trainer.test(showFigure=True)

if __name__ == "__main__":
  config, unparsed = get_config()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
