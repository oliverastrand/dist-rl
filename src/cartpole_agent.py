import gym
import numpy as np
import tensorflow as tf

from agent import Agent
from cartpole_learner import CartPoleLearner as Learner


class CartPoleAgent(Agent):

    # Returns a Cart Pole environment and observation and action spaces
    def make_env(self):
        env = gym.make('CartPole-v0')
        return env, env.observation_space.shape, env.action_space.n

    # Returns a new Learner
    def make_learner(self):
        return Learner(observation_shape=self.observation_space_shape,
                       nr_actions=self.nr_actions)

    # Returns the state, which is just the observation vector
    def make_next_state(self, observation):
        return observation

    # Returns the first state of an episode
    def make_first_state(self):
        return self.env.reset()

# This class can be tested easily as below
if __name__ == "__main__":
    a = CartPoleAgent()
    sess = tf.Session()
    a.init_params(sess)
    a.run_episodes(sess)
