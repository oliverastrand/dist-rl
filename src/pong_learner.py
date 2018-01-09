import tensorflow as tf

from src.learner import Learner


class PongLearner(Learner):

    def __init__(self, observation_shape, nr_actions, alpha=0.01):

        # Save parameters unique to Pong
        self.conv1_size = 10
        self.conv1_filters = 32
        self.conv1_stride = [1, 4, 4, 4, 1]

        self.conv2_size = 4
        self.conv2_filters = 64
        self.conv2_stride = [1, 4, 4, 4, 1]

        self.conv3_size = 3
        self.conv3_filters = 64
        self.conv3_stride = [1, 2, 2, 2, 1]

        self.FCN_size = 256

        self.pkeep = 0.75
        self.kernel_size = [1, 1, 3, 3, 1]
        self.pool_strides = [1, 3, 3, 3, 1]

        super(PongLearner, self).__init__(observation_shape, nr_actions, alpha)

    # Create the tf Graph for the Deep Neural Network
    def init_neural_net(self, input_values):

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv3d(x, W, stride):
            return tf.nn.conv3d(x, W, strides=stride, padding='SAME')

        def pool(h):
            return tf.nn.max_pool3d(h,self.kernel_size,self.pool_strides, padding = 'SAME')


        # Convolutional Layers
        w_conv1 = weight_variable((self.conv1_size, self.conv1_size, 3, 4, self.conv1_filters), "Weights1")
        b_conv1 = bias_variable([self.conv1_filters], "Bias1")

        w_conv2 = weight_variable((self.conv2_size, self.conv2_size, 3,
                                   self.conv1_filters, self.conv2_filters), "Weights2")
        b_conv2 = bias_variable([self.conv2_filters], "Bias2")

        w_conv3 = weight_variable((self.conv3_size, self.conv3_size, 3,
                                   self.conv2_filters, self.conv3_filters), "Weights3")
        b_conv3 = bias_variable([self.conv3_filters], "Bias3")

        h_conv1 = pool(tf.nn.relu(conv3d(input_values, w_conv1, self.conv1_stride) + b_conv1))
        h_conv2 = pool(tf.nn.relu(conv3d(h_conv1, w_conv2, self.conv2_stride) + b_conv2))
        h_conv3 = pool(tf.nn.relu(conv3d(h_conv2, w_conv3, self.conv3_stride) + b_conv3))

        # Fully Connected Layers
        flat = tf.reshape(h_conv3, [-1, 64])
        w_full = weight_variable([64, self.FCN_size], 'Weights4')
        b_full = bias_variable([self.FCN_size], "Bias4")

        dense = tf.matmul(flat, w_full)
        dense = tf.nn.relu(dense+b_full)
        densed = tf.nn.dropout(dense, self.pkeep)

        # Final Output Layers
        w_final = weight_variable([self.FCN_size, self.nr_actions], "Weights5")
        b_final = bias_variable([self.nr_actions], "Bias5")
        ylogits = tf.matmul(densed, w_final) + b_final

        weight_list = [w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_full, b_full,w_final, b_final]

        return ylogits, weight_list
