import tensorflow as tf

class Learner:

    def __init__(self, options):

        # We need to have a reference to the memory
        self.memory = None

        # This placeholder will be filled with experiences from the memory
        self.input_state = tf.placeholder(tf.float32, shape=(None, options["env_size"]))

        # We generate one network that will be updated during training
        # and one fixed network that will be the target values (updated periodically)
        self.out_layer = self.initialize_network(options["DNN"])
        self.target_out_layer = self.initialize_network(options["DNN"])

        self.targets = tf.placeholder(tf.float32, shape=(None, options["env_actions"]))
        self.loss = self.calc_loss(self.out_layer, self.targets, options["loss"])

        # Get our tensorflow session
        self.sess = tf.Session()

    def predict(self, state):
        """
        Function that returns the predicted Q-value of each action for a given state
        :param state:
        :return:
        """
        return self.sess.run(self.out_layer, feed_dict={"self.input_state": state})

    def predict_targets(self, state):
        """
        Function that returns the predicted Q-value of the target network for
        each action for a given state
        :param state:
        :return:
        """
        return self.sess.run(self.target_out_layer, feed_dict={"self.input_state": state})


    def calc_loss(self, outputs, targets, options):
        """
        Calculate the loss from outputs and targets, with given options
        :param outputs:
        :param targets:
        :param options:
        :return:
        """
        pass

    def calc_gradient(self, states, targets):
        """
        Function that calculates the gradient of the loss function w.r.t. the NN weights
        :param states:
        :param targets:
        :return:
        """
        pass

    def update_regular_network(self, weights):
        """
        Method that updates the regular network
        :param weights:
        :return:
        """
        pass

    def update_target_network(self, weights):
        """
        Method that updates the target network
        :param weights:
        :return:
        """
        pass

    def initialize_network(self, options):
        """
        Initializes a neural network with the given options
        :param options:
        :return: The final layer of the network
        """
        pass