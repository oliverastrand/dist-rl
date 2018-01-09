import tensorflow as tf

from learner import Learner


class CartPoleLearner(Learner):

    def __init__(self, observation_shape, nr_actions, alpha=0.01):

        # Save parameters unique to Cart Pole
        self.nr_hidden_1 = 24
        self.nr_hidden_2 = 48

        super(CartPoleLearner, self).__init__(observation_shape, nr_actions, alpha)

    # Create the tf Graph for the Deep Neural Network
    def init_neural_net(self, input_values):

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        bias1 = bias_variable((self.nr_hidden_1,), "Bias1")
        wei1 = weight_variable(self.observation_shape + (self.nr_hidden_1,), "Weights1")
        bias2 = bias_variable((self.nr_hidden_2,), "Bias2")
        wei2 = weight_variable((self.nr_hidden_1, self.nr_hidden_2), "Weights2")
        bias3 = bias_variable((self.nr_actions,), "Bias3")
        wei3 = weight_variable((self.nr_hidden_2, self.nr_actions), "Weights3")

        weight_list = [wei1, bias1, wei2, bias2, wei3, bias3]

        hidden_1 = tf.nn.tanh(tf.matmul(input_values, wei1) + bias1)
        hidden_2 = tf.nn.tanh(tf.matmul(hidden_1, wei2) + bias2)
        out_layer = tf.matmul(hidden_2, wei3) + bias3

        return out_layer, weight_list
