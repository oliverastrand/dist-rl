import tensorflow as tf

class Learner:

    def __init__(self, options):

        # We need to have a reference to the memory
        self.memory = None

        # This placeholder will be filled with experiences from the memory
        self.input_state = tf.placeholder(tf.float32, shape=(None, options["env_size"][0], options["env_size"][1]))

        # We generate one network that will be updated during training
        # and one fixed network that will be the target values (updated periodically)
        self.out_layer, self.weight_list = self.initialize_network(options["DNN"])
        self.target_out_layer, self.target_weight_list = self.initialize_network(options["DNN"])

        self.targets = tf.placeholder(tf.float32, shape=(None, options["env_actions"]))
        self.loss = self.calc_loss(self.out_layer, self.targets, options["loss"])

        init_op = tf.global_variables_initializer()

        # Get our tensorflow session and initialize our variables
        self.sess = tf.Session()
        self.sess.run(init_op)

    def predict(self, state):
        """
        Function that returns the predicted Q-value of each action for a given state
        """

        return self.sess.run(self.out_layer, feed_dict={self.input_state: state})

    def predict_targets(self, state):
        """
        Function that returns the predicted Q-value of the target network for
        each action for a given state
        """
        return self.sess.run(self.target_out_layer, feed_dict={self.input_state: state})


    def calc_loss(self, outputs, targets, options):
        """
        Calculate the loss from outputs and targets, with given options
        """
        return tf.losses.mean_squared_error(targets, outputs)

    def calc_gradient(self, states, targets):
        """
        Function that calculates the gradient of the loss function w.r.t. the NN weights
        """
        loss = self.calc_loss(self.out_layer, targets, {})
        gradients = tf.gradients(self.loss, self.weight_list)
        print(states)
        print(gradients)
        print(self.loss)
        return self.sess.run(gradients, feed_dict={self.input_state: states, self.targets: targets})

    def update_regular_network(self, weights):
        """
        Method that updates the regular network
        """
        pass

    def update_target_network(self, weights):
        """
        Method that updates the target network
        """
        for i in range(len(weights)):
            assign_op = self.target_weight_list[i].assign(weights[i])
            self.sess.run(assign_op)

    def initialize_network(self, options):
        """
        Initializes a neural network with the given options
        """
        nr_hidden = 5
        input_size = 4 * 2
        nr_outputs = 3

        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        bias1 = bias_variable((nr_hidden,))
        wei1 = weight_variable((input_size, nr_hidden))
        bias2 = bias_variable((nr_outputs,))
        wei2 = weight_variable((nr_hidden, nr_outputs))

        weight_list = []
        weight_list.append(wei1)
        weight_list.append(bias1)
        weight_list.append(wei2)
        weight_list.append(bias2)

        input_flat = tf.reshape(self.input_state, (-1, input_size))
        hidden = tf.nn.relu(tf.matmul(input_flat, wei1) + bias1)
        out_layer = tf.matmul(hidden, wei2) + bias2

        return out_layer, weight_list

    # Temporary method for testing the algorithm
    def train(self, states, targets):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_opt = optimizer.minimize(self.loss)

        self.sess.run([train_opt], feed_dict={self.input_state: states, self.targets: targets})

