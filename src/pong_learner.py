import tensorflow as tf


class Learner:

    def __init__(self, observation_shape, nr_actions, alpha=0.01):

        self.alpha = alpha
        self.nr_hidden_1 = 24
        self.nr_hidden_2 = 48

        self.observation_shape = observation_shape
        self.nr_actions = nr_actions

        self.states = tf.placeholder(dtype=tf.float32, shape=(None,) + observation_shape)
        self.targets = tf.placeholder(dtype=tf.float32, shape=(None, nr_actions))

        self.predictions, self.weight_list = self.init_neural_net(self.states)

        loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = optimizer.minimize(loss=loss)

        self.gradients_op = optimizer.compute_gradients(loss, self.weight_list)

        self.grads = tf.placeholder(dtype=tf.float32)
        self.vars = tf.placeholder(dtype=tf.float32)
        self.apply_gradients_op = optimizer.apply_gradients(self.gradients_op)

        self.target_predictions, self.target_weight_list = self.init_neural_net(self.states)

        self.init_op = tf.global_variables_initializer()

    def init_params(self, sess):
        sess.run(self.init_op)

    def predict(self, state, sess):
        return sess.run(self.predictions, feed_dict={self.states: state})

    def predict_targets(self, state, sess):
        return sess.run(self.target_predictions, feed_dict={self.states: state})

    def fit(self, states, targets, sess):
        sess.run(self.train_op, feed_dict={self.states: states, self.targets: targets})

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

    def get_weights(self):
        return self.weight_list
