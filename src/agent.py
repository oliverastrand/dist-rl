import gym
import numpy as np
import random
import tensorflow as tf

from abc import ABC, abstractmethod
from collections import deque


class Agent(ABC):
    def __init__(self, task=None, reward_decay=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.99, batch_size=256, tensorboard=False, prefill_length=1000):

        # Stores all the parameters of the agent
        self.memory = deque(maxlen=10000)
        self.reward_decay = reward_decay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        if tensorboard:
            self.tensorboard = True
            self.writer = tf.summary.FileWriter('/tmp/pong')
        else:
            self.tensorboard = False

        if task is None:
            self.task = 1
        else:
            self.task = task

        self.env, self.observation_space_shape, self.nr_actions = self.make_env()

        self.learner = self.make_learner()
        self.prefill_length = prefill_length

    # Method that returns an environment, the shape of the states, and the action space
    @abstractmethod
    def make_env(self):
        return None, None, None

    # Method that returns a new Learner object
    @abstractmethod
    def make_learner(self):
        pass

    # Initializes the TF Graph
    def init_params(self, sess: tf.Session):
        self.learner.init_params(sess)

    # Chooses an action epsilon greedily using predicted Q-values
    def choose_action(self, state, epsilon, sess):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.learner.predict(np.expand_dims(state, axis=0), sess))

    # Returns the current value of epsilon
    def get_epsilon(self):
        return max(self.epsilon_min, self.epsilon)

    # Method that loads experiences from memory and trains on them
    def replay(self, batch_size, sess):
        x_batch, y_batch = [], []

        # Randomly sample from memory
        batch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        # Un-pack each loaded experience and calculate target values for training
        for state, action, reward, next_state, done in batch:
            q_target = self.learner.predict(np.expand_dims(state, axis=0), sess)[0, :]

            # Updates target Q-value according to Bellman Eq.
            if done:
                q_target[action] = reward
            else:
                q_target[action] = reward + self.reward_decay * np.max(
                    self.learner.predict(np.expand_dims(next_state, axis=0), sess)[0, :])

            x_batch.append(state)
            y_batch.append(q_target)

        # Train the neural network
        self.learner.fit(np.array(x_batch), np.array(y_batch), sess)

    # Store an experience in the memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Method that makes state out of an observation
    @abstractmethod
    def make_next_state(self, observation):
        pass

    # Method that makes the first state
    @abstractmethod
    def make_first_state(self):
        pass

    # Method that fills the memory with experiences
    def prefill_memory(self, sess):

        state = self.make_first_state()
        done = False

        while len(self.memory) < self.prefill_length:
            if done:
                state = self.make_first_state()

            # Choose random action and observe result
            action = self.choose_action(state, 1, sess)
            frame, reward, done, _ = self.env.step(action)
            next_state = self.make_next_state(frame)

            # Save the observed experience
            self.remember(state, action, reward, next_state, done)

            state = next_state

    # Runs and trains the agent for a set number of episodes
    def run_episodes(self, sess, n_episodes=1000):

        # Saves points for recording performance
        points = deque(maxlen=100)

        self.prefill_memory(sess)

        # Repeats same procedure each episode
        for e in range(n_episodes):

            state = self.make_first_state()
            done = False
            i = 0

            # While the game is not over
            while not done:

                # Interact with the environment and observe result
                action = self.choose_action(state, self.get_epsilon(), sess)
                frame, reward, done, _ = self.env.step(action)
                next_state = self.make_next_state(frame)
                self.remember(state, action, reward, next_state, done)
                state = next_state

                i += reward

            points.append(i)

            # Load from memory and train the learner
            self.replay(self.batch_size, sess)

            if e % 100 == 0:
                print('[Episode {}] - Score acquired by our Space Gorila: {}.'.format(e, np.mean(points)))

            if self.tensorboard:
                summary = tf.Summary(value=[tf.Summary.Value(tag='score_{}'.format(self.task), simple_value=i)])
                self.writer.add_summary(summary, e)

            # Deprecate the chance of choosing random action
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
