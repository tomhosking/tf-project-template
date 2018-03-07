import tensorflow as tf
import numpy as np

class TFModel():
    def __init__(self):
        self.saver = tf.train.Saver()
        self.is_training = tf.placeholder_with_default(False,(), "is_training")
        self._train_summaries=[]
        self._eval_summaries=[]

        self.build_model()

        self._train_summaries.extend([self.loss(), self.accuracy()])
        self._eval_summaries.extend([self.loss(), self.accuracy()])
        self.train_summary = tf.merge_summary(self._train_summaries)
        self.eval_summary = tf.merge_summary(self._eval_summaries)

    # FILL THIS METHOD IN
    def build_model(self):
        pass

    def init():
        if self.to_restore:
            return
        else:
            return tf.global_variables_initializer()


    def loss(self):
        return self.loss

    def optimizer(self):
        return self.opt

    def placeholders(self):
        return self.x, self.y, self.is_training

    def accuracy(self):
        return self.accuracy

    def predict(self):
        return self.y_hat
