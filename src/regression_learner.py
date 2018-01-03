import tensorflow as tf
import numpy as np


class RegressionLearner:

    def __init__(self):

        self.k = tf.Variable(0.0, dtype=tf.float32)
        self.m = tf.Variable(0.0, dtype=tf.float32)

        self.x = tf.placeholder(dtype=tf.float32, shape=[None])
        self.y = tf.scalar_mul(self.k, self.x) + self.m

        self.y_hat = tf.placeholder(dtype=tf.float32, shape=[None])

        self.loss = tf.losses.mean_squared_error(self.y_hat, self.y)

        self.optimizer = tf.train.AdamOptimizer(0.01)
        self.train_op = self.optimizer.minimize(loss=self.loss)

        self.init_op = tf.global_variables_initializer()

        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.step_op = self.global_step_tensor.assign_add(1)

    def init_params(self, session):
        session.run(self.init_op)

    def train(self, x, y, session):
        _, _, k, m = session.run([self.train_op, self.step_op, self.k, self.m], feed_dict={self.x: x, self.y_hat: y})
        print([k,m])
