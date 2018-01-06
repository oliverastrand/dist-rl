import numpy as np
import random
import tensorflow as tf

from collections import deque

from regression_learner import RegressionLearner


class RegressionAgent:

    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.k = 3
        self.m = -5

        self.learner = RegressionLearner()

    def init_params(self, session):
        self.learner.init_params(session)

    def gen_data(self, n=100):
        X = np.random.normal(0, 10, 100)
        epsilon = np.random.normal(0, 0.01)
        Y = self.k * X + self.m + epsilon
        return X, Y

    def save_data(self, x, y):

        for i in range(len(x)):
            self.memory.append((x[i], y[i]))

    def replay(self, session, batch_size=100):
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        x_data = []
        y_data = []
        for x, y in minibatch:
            x_data.append(x)
            y_data.append(y)

        self.learner.train(x_data, y_data, session)

    def play_and_train_epoch(self, session):
        x, y = self.gen_data()
        self.save_data(x, y)
        self.replay(session, 100)


