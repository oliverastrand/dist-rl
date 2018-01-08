import gym
import math
import numpy as np
import random
import requests
import tensorflow as tf
import time

from collections import deque

from pong_learner import Learner

class Agent:

    def __init__(self, task, n_win_ticks=195, gamma=1.0, epsilon=1.0, epsilon_min=0.01,
                 epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=256, monitor=False):
        self.memory = deque(maxlen=10000)
        self.env = gym.make('Pong-v0')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.writer = tf.summary.FileWriter('/tmp/pong')
        self.task = task
        
        self.observation_space_shape = (210,160,3,4)#self.env.observation_space.shape
        self.nr_actions = self.env.action_space.n

        self.learner = Learner(observation_shape=self.observation_space_shape,
                               nr_actions=self.nr_actions)

        self.frame_queue = deque(maxlen=4)
        self.prefill_length = 1000

    def init_params(self, sess: tf.Session):
        self.learner.init_params(sess)

    def choose_action(self, state, epsilon, sess):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.learner.predict(np.expand_dims(state, axis=0), sess))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def replay(self, batch_size, sess):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.learner.predict(np.expand_dims(state, axis=0), sess)[0, :]
            y_target[action] = reward if done else reward + self.gamma * np.max(
                self.learner.predict(np.expand_dims(next_state, axis=0), sess)[0, :])
            x_batch.append(state)
            y_batch.append(y_target)

        self.learner.fit(np.array(x_batch), np.array(y_batch), sess)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def make_next_state(self, frame):
        self.frame_queue.append(frame)
        tensor = np.zeros(shape=(210, 160, 3, 4))
        for i in range(4):
            tensor[:, :, :, i] = self.frame_queue.pop()
            self.frame_queue.appendleft(tensor[:, :, :, i])

        return tensor

    def make_first_state(self):
        frame = self.env.reset()
        self.frame_queue.append(frame)
        for i in range(2):
            frame, _, _, _ = self.env.step(self.env.action_space.sample())
            self.frame_queue.append(frame)
        frame, _, _, _ = self.env.step(self.env.action_space.sample())
        return self.make_next_state(frame)
    
    def prefill_memory(self, sess):
        state = self.make_first_state()
        while len(memory)<self.prefill_length:
            done = False
            while not done or length(memory)<self.prefill_length:
                action = self.choose_action(state, 1, sess)
                frame, reward, done, _ = self.env.step(action)
                next_state = self.make_next_state(frame)
                self.remember(state, action, reward, next_state, done)
                state = next_state

    def run_episodes(self, sess, n_episodes=1000):
        scores = deque(maxlen=100)
        self.prefill_memory(sess)
        for e in range(n_episodes):
            state = self.make_first_state()
            done = False
            i = 0
            while not done:
                action = self.choose_action(state, self.get_epsilon(e), sess)
                frame, reward, done, _ = self.env.step(action)
                next_state = self.make_next_state(frame)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                i += reward

            print('[Episode {}] - Score acquired by our Space Gorila: {}.'.format(e, i))

            summary = tf.Summary(value=[tf.Summary.Value(tag=f'score_{self.task}', simple_value=i)])
            self.writer.add_summary(summary, e)

            self.replay(self.batch_size, sess)


if __name__ == "__main__":
    a = Agent()
    sess = tf.Session()
    a.init_params(sess)
    a.run_episodes(sess)
