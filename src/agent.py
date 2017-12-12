import gym
import numpy as np

class Agent:

    def __init__(self, learner, environment: gym.ActionWrapper):
        self.learner = learner
        self.environment = environment

        self.memory = None # Save reference to memory
        self.parameter_server = None # Save reference to parameter server
        self.state = self.environment.reset()

        self.epsilon = 0.05
        self.batch_size = 100

    def step(self, update_target: bool):
        """
        The main body of the algorithm 1 in the Gorila paper.
        The agent interacts with the environment, saves the experience,
        updates its learners network, calculates and returns gradients,
        and every N steps updates its learners target network parameters
        :return:
        """
        self.act()
        self.learner.update_regular_network(self.parameter_server.get_weights())

        states, targets = self.get_data_batch()
        gradient = self.learner.calc_gradient(states, targets)

        self.parameter_server.send_gradient(gradient)

        if update_target:
            self.learner.update_target_network(self.parameter_server.get_weights())

    def act(self):
        """
        Method that makes agent choose best action epsilon greedily, and interact with environment
        """
        action = self.choose_action(self.state, self.epsilon)
        new_state, reward, done, info = self.environment.step(action)

        # Save the experience in the memory
        self.memory.save(self.state, action, reward, new_state)

        self.state = new_state

    def choose_action(self, state, epsilon):
        """
        Function that returns highest Q-value action with probability 1 - epsilon, otherwise random
        :param state:
        :param epsilon:
        :return:
        """
        action_values = self.learner.predict(state)
        if np.random() < epsilon:
            action = None # Should choose random action
        else:
            action = np.argmax(action_values)
        return action

    def save_data(self, data):
        """
        Method that pushes some experience to the memory
        :param data:
        :return:
        """
        pass

    def get_data_batch(self):
        """
        Function that retrieves a batch of experiences from memory.
        Then uses the Bellman equation to calculate target values,
        and returns a data set ready for training
        :return:
        """
        pass
