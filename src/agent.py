import gym
import numpy as np

from memory import Memory

class Agent:

    def __init__(
        self,
        learner,
        environment: gym.ActionWrapper,
        epsilon=0.1,
        nr_actions = 3,
        nr_frames = 4,
        batch_size = 10,
        learning_rate = 0.1,
        memory_size = 100,
    ):
        self.nr_actions = nr_actions
        self.nr_frames = nr_frames
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_size = memory_size

        self.learner = learner
        self.environment = environment
        self.state = self.initialize_state()

        self.memory = Memory(self.memory_size) # Save reference to memory
        self.server_parameters = self.learner.weight_list # Save reference to parameter server.
        # The parameter server object currently needs get_weights and send_gradients methods

        self.populate_memory()

    def step(self, update_target: bool):
        """
        The main body of Algorithm 1 in the Gorila paper.
        The agent interacts with the environment, saves the experience,
        updates its learners network, calculates and returns gradients,
        and every N steps updates its learners target network parameters
        """
        self.act()
        self.learner.update_regular_network(self.get_server_parameters())

        states, targets = self.get_data_batch()

        # temporary remove for testing
        #gradient = self.learner.calc_gradient(states.reshape((self.batch_size, 2, self.nr_frames)), targets)
        #self.update_server_parameters(gradient)

        self.learner.train(states.reshape((self.batch_size, 2, self.nr_frames)), targets)
        self.server_parameters = self.learner.weight_list

        if update_target:
            self.learner.update_target_network(self.get_server_parameters())

    def act(self):
        """
        Method that makes agent choose best action epsilon greedily, and interact with environment
        """
        action = self.choose_action(self.state, self.epsilon)
        observation, reward, done, info = self.environment.step(action)

        print(action, reward, observation)

        new_state = self.make_new_state(self.state, observation)

        # Save the experience in the memory
        self.memory.save((self.state, action, reward, new_state))

        self.state = new_state

    def choose_action(self, state, epsilon):
        """
        Function that returns highest Q-value action with probability 1 - epsilon, otherwise random
        """
        if np.random.sample() < epsilon:
            action = self.environment.action_space.sample() # Should choose random action
        else:
            action_values = self.learner.predict(np.expand_dims(state, axis=0))
            action = np.argmax(action_values)
        return action

    def save_data(self, data):
        """
        Method that pushes some experience to the memory
        """
        pass

    def get_data_batch(self):
        """
        Function that retrieves a batch of experiences from memory.
        Then uses the Bellman equation to calculate target values,
        and returns a data set ready for training
        """
        raw_experiences = self.memory.get_random_sample(self.batch_size)
        # The state is represented by the 2 values at each time step,
        # times the 4 frames that are analyzed at a time
        ret_states = np.zeros((self.batch_size, 2 * self.nr_frames))
        ret_targets = np.zeros((self.batch_size, self.nr_actions))
        for i in range(self.batch_size):
            state = raw_experiences[i][0]
            action = raw_experiences[i][1]
            reward = raw_experiences[i][2]
            new_state = raw_experiences[i][3]
            new_q = reward + 0.99 * np.max(self.learner.predict_targets(np.expand_dims(new_state, axis=0)))
            ret_states[i, :] = state.flatten()
            ret_targets[i, :] = self.learner.predict_targets(np.expand_dims(state, axis=0))
            ret_targets[i, action] = new_q

        return ret_states, ret_targets

    def make_new_state(self, old_state, new_observation):
        ret = np.zeros(old_state.shape)
        ret[:, :-1] = old_state[:, 1:]
        ret[:, -1] = new_observation
        return ret

    def initialize_state(self):
        first_state = np.zeros((2, self.nr_frames))
        first_state[:, 0] = self.environment.reset()
        for i in range(1, self.nr_frames):
            first_state[:, i], _, _, _ = self.environment.step(0)
        return first_state

    def populate_memory(self):
        for i in range(self.memory_size):
            self.act()

    # Functions that should belong to the parameter server
    def update_server_parameters(self, gradient):
        for i in range(len(self.server_parameters)):
            self.server_parameters[i] = self.server_parameters[i] - \
                                        self.learning_rate * gradient[i]
    def get_server_parameters(self):
        return self.server_parameters
