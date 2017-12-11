import gym

class Agent:

    def __init__(self, learner, environment: gym.ActionWrapper):
        self.learner = learner
        self.environment = environment

        self.state = self.environment.reset()


    def act(self):
        """
        Method that makes agent choose best action epsilon greedily, and interact with environment
        """
        pass

    def choose_action(self, state, epsilon):
        """
        Function that returns highest Q-value action with probability 1 - epsilon, otherwise random
        :param state:
        :param epsilon:
        :return:
        """
        pass

    def save_data(self, data):
        """
        Method that pushes some experience to the memory
        :param data:
        :return:
        """
        pass
