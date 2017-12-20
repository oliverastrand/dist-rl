import tensorflow as tf

class Learner():

    def __init__(self, alpha=0.01):

        self.alpha = alpha
        self.nr_hidden_1 = 24
        self.nr_hidden_2 = 48
        self.input_size = 4
        self.output_size = 2

        self.states = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

        self.predictions, self.weight_list = self.init_neural_net(self.states)

        loss = tf.losses.mean_squared_error(labels=self.targets, predictions=self.predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = optimizer.minimize(loss=loss)

        self.gradients_op = optimizer.compute_gradients(loss, self.weight_list)

        self.grads = tf.placeholder(dtype=tf.float32)
        self.vars = tf.placeholder(dtype=tf.float32)
        self.apply_gradients_op = optimizer.apply_gradients(self.gradients_op)

        self.target_predictions, self.target_weight_list = self.init_neural_net(self.states)

        #self.new_weights = tf.placeholder(dtype=tf.float32, shape=None)
        #self.update_weights_ops = self.get_update_weights_ops(self.new_weights)

        self.save_dict = self.get_save_dict(self.weight_list)
        self.target_save_dict = self.get_save_dict(self.target_weight_list)

        self.weights_saver = tf.train.Saver(self.save_dict)
        self.target_weights_saver = tf.train.Saver(self.target_save_dict)

        self.sess = tf.Session()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def predict(self, state):
        return self.sess.run(self.predictions, feed_dict={self.states: state})

    def predict_targets(self, state):
        return self.sess.run(self.target_predictions, feed_dict={self.states: state})

    def fit(self, states, targets):
        self.sess.run(self.train_op, feed_dict={self.states: states, self.targets: targets})

    def gradients_fit(self, states, targets):
        self.sess.run(self.apply_gradients_op, feed_dict={self.states: states, self.targets: targets})

    def update_targets_weights(self, path):
        """
        copy_ops = []
        for i in range(len(weight_list)):
            copy_ops.append(self.target_weight_list[i].assign(weight_list[i]))
        self.sess.run(copy_ops)
        """
        self.target_weights_saver.restore(self.sess, path)

    #def get_update_weights_ops(self, new_weights):
    #    update_ops = []
    #    for i in range(len(self.weight_list)):
    #        update_ops.append(self.weight_list[i].assign(new_weights))
    #    return update_ops

    def update_weights(self, path):
        """
        for i in range(len(new_weight_list)):
            print(new_weight_list[i])
            self.sess.run(self.update_weights_ops[i], feed_dict={self.new_weights: new_weight_list[i]})
        """
        self.weights_saver.restore(self.sess, path)

    def get_save_dict(self, li):
        d = {}
        for i in range(len(li)):
            d[str(i)] = li[i]
        return d

    def save_weights(self, path):
        self.weights_saver.save(self.sess, path)

    def get_gradients(self, states, targets):
        return self.sess.run(self.gradients_op, feed_dict={self.states: states, self.targets: targets})

    def apply_gradients(self, grads_and_vars):
        self.sess.run(self.apply_gradients_op,
                      feed_dict={self.grads: grads_and_vars[0], self.vars: grads_and_vars[1]})

    def init_neural_net(self, input_values):

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        bias1 = bias_variable((self.nr_hidden_1,), "Bias1")
        wei1 = weight_variable((self.input_size, self.nr_hidden_1), "Weights1")
        bias2 = bias_variable((self.nr_hidden_2,), "Bias2")
        wei2 = weight_variable((self.nr_hidden_1, self.nr_hidden_2), "Weights2")
        bias3 = bias_variable((self.output_size,), "Bias3")
        wei3 = weight_variable((self.nr_hidden_2, self.output_size), "Weights3")

        weight_list = []
        weight_list.append(wei1)
        weight_list.append(bias1)
        weight_list.append(wei2)
        weight_list.append(bias2)
        weight_list.append(wei3)
        weight_list.append(bias3)

        hidden_1 = tf.nn.tanh(tf.matmul(input_values, wei1) + bias1)
        hidden_2 = tf.nn.tanh(tf.matmul(hidden_1, wei2) + bias2)
        out_layer = tf.matmul(hidden_2, wei3) + bias3

        return out_layer, weight_list
        #hid1 = tf.layers.dense(inputs=input_values, units=24, activation=tf.nn.tanh)
        #hid2 = tf.layers.dense(inputs=hid1, units=48, activation=tf.nn.tanh)
        #return tf.layers.dense(inputs=hid2, units=2)

    def get_weights(self):
        return self.weight_list
