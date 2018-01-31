import os
import numpy as np

import tensorflow as tf

class Trainer(object):
  def __init__(self, config, model, rng):
    self.config = config
    self.rng = rng

    self.model_dir = config.log_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step

    self.model = model

    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)


  def train(self):
    print("[*] Training starts...")
    self.model.train()


  def test(self, showFigure = False):
    self.model.test(weights_file=self.config.weights_path,
                    showFigure=showFigure)

  def resume_training(self):
    print("resuming training from weights file: ", self.config.weights_path)
    self.model.resume_training(self.config.weights_path, 50)