import tensorflow as tf


class Learner:

    def __init__(self, observation_shape, nr_actions, alpha=0.01):

        self.alpha = alpha
        self.conv1_size = 10
        self.conv1_filters = 32
        self.conv1_stride = [1, 5, 5, 1]

        self.conv2_size = 4
        self.conv2_filters = 64
        self.conv2_stride = [1, 2, 2, 1]

        self.conv3_size = 3
        self.conv3_filters = 64
        self.conv3_stride = [1, 1, 1, 1]

        self.FCN_size = 256

        self.pkeep = 0.75
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

        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

        #CONVOLUTIONAL LAYERS
        w_conv1 = weight_variable((self.conv1_size, self.conv1_size, 3,self.conv1_filters), "Weights1")
        b_conv1 = bias_variable([self.conv1_filters],"Bias1")

        w_conv2 = weight_variable((self.conv2_size, self.conv2_size, self.conv1_filters,self.conv2_filters), "Weights2")
        b_conv2 = bias_variable([self.conv2_filters],"Bias2")

        w_conv3 = weight_variable((self.conv3_size, self.conv3_size, self.conv2_filters,self.conv3_filters), "Weights3")
        b_conv3 = bias_variable([self.conv3_filters],"Bias3")

        h_conv1 = tf.nn.relu(conv2d(input_values, w_conv1, self.conv1_stride) + b_conv1  )
        h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2, self.conv2_stride) + b_conv2  )
        h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3, self.conv3_stride) + b_conv3  )

        #FULLY CONNECTED LAYER
        flat = tf.reshape(h_conv3, [-1, 21 * 16 * self.conv3_filters])
        w_full = weight_variable([21 * 16 * self.conv3_filters, self.FCN_size],'Weights4')
        b_full = bias_variable([self.FCN_size],"Bias4")

        dense = tf.matmul(flat, w_full)
        dense = tf.nn.relu(dense+b_full)
        densed = tf.nn.dropout(dense,self.pkeep)

        #FINAL OUTPUT LAYER
        w_final = weight_variable([self.FCN_size, self.nr_actions],"Weights5")
        b_final = bias_variable([self.nr_actions],"Bias5")

        Ylogits = tf.matmul(densed,w_final) + b_final
        weight_list = [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_full, b_full,w_final, b_final]

        return Ylogits, weight_list

    def get_weights(self):
        return self.weight_list
