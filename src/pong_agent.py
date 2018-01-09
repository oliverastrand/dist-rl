import gym
import numpy as np
import tensorflow as tf

from collections import deque

from src.agent import Agent
from src.pong_learner import Learner


class PongAgent(Agent):

    def __init__(self, task=None, reward_decay=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.99, batch_size=256):

        # Creates a queue of frames to capture movement in the state
        self.frame_depth = 4
        self.frame_queue = deque(maxlen=self.frame_depth)

        super(PongAgent, self).__init__(task, reward_decay, epsilon, epsilon_min, epsilon_decay, batch_size)

    # Return the pong environment and state space and action space shapes
    def make_env(self):
        env = gym.make('Pong-v0')
        return env, (210, 160, 3, self.frame_depth), env.action_space.n

    # Return a Learner object
    def make_learner(self):
        return Learner(observation_shape=self.observation_space_shape,
                       nr_actions=self.nr_actions)

    # Makes a state out of the last couple of frames (observations)
    def make_next_state(self, observation):

        # Adds the new observation
        self.frame_queue.append(observation)
        state_tensor = np.zeros(shape=self.observation_space_shape)

        # Adds each of the frames in the queue to the state tensor
        for i in range(self.frame_depth):
            state_tensor[:, :, :, i] = self.frame_queue.pop()
            self.frame_queue.appendleft(state_tensor[:, :, :, i])

        return state_tensor

    # Acts a few times so that the frame queue gets filled up
    def make_first_state(self):
        frame = self.env.reset()
        self.frame_queue.append(frame)
        for i in range(self.frame_depth - 2):
            frame, _, _, _ = self.env.step(self.env.action_space.sample())
            self.frame_queue.append(frame)
        frame, _, _, _ = self.env.step(self.env.action_space.sample())
        return self.make_next_state(frame)

# This class can be tested easily as below
if __name__ == "__main__":
    a = PongAgent()
    sess = tf.Session()
    a.init_params(sess)
    a.run_episodes(sess)
