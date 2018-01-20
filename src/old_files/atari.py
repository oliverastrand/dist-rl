%matplotlib inline
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from IPython import display

def choose_action(observation):
    return np.random.randint(0, n_actions)

def store_transition(s, a, r, s_):
    memory.append([s, a, r, s_])

def create_Wb(shape, name, trainable):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=f'W_{name}')
    b = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name=f'b_{name}')
    return W, b

def build_net(shape, name, trainable):

    x = tf.placeholder(tf.uint8, [None, 210, 160, 3, 4], name='state')
    print(x)
    x = tf.resize_images(x, [84, 84], align_corners=True)
    print(x)
    x = tf.to_float(x) / 255.0
    print(x)

    with tf.variable_scope(f'conv1_{name}'):
        W_conv1, b_conv1 = create_Wb([8, 8, 4, 32], trainable, 'conv1')
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name='h_conv1')
        print(W_conv1, b_conv1, h_conv1)

    with tf.variable_scope(f'conv2_{name}'):
        W_conv2, b_conv2 = create_Wb([4, 4, 32, 64], trainable, 'conv2')
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv1, name='h_conv2')
        print(W_conv2, b_conv2, h_conv2)

    h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64], name='h_conv2_flat')
    print(h_conv2_flat)

    with tf.variable_score(f'fc1_{name}'):
        W_fc1, b_fc1 = create_Wb([7 * 7 * 64, n_actions], trainable, 'fc1')
        print(W_fc1, b_fc1)
        y = tf.nn.relu(h_conv2_flat @ W_fc1 + b_fc1, name='h_fc1')
        print(y)

    return x, y

def init():
    sess = tf.Session()

    self.summaryMerged = tf.summary.merge_all()

    x, y = build_net('policy', True)
    x_target, y_target = build_net('target', False)
    replace_target_ops = []
    trainable_vars = tf.trainable_variables()
    all_variables = tf.global_variables()

    for i in range(0, len(trainable_vars)):
        update_target_ops.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

    a = tf.placeholder(tf.float32, shape=[None, n_actions])
    y_ = tf.placeholder(tf.float32, [None])
    y_a = tf.reduce_sum(y@a, reduction_indices=1)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(y_a, y_))

    with tf.variable_scope('train'):
        train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)

    sess.run(tf.global_variables_initializer())

def run():
    if counter % replace_target_iter == 0:
        sess.run(replace_target_ops)

    if memory_counter > memory_size:
        sample_index = np.random.choice(memory_size, size=batch_size)
    else:
        sample_index = np.random.choice(memory_counter, size=batch_size)
    batch_memory = memory[sample_index, :]
    y, y_ = sess.run([y, y_], feed_dict={self.s_: batch_memory[-1], self.s: batch_memory[0]})

img = plt.imshow(env.render(mode='rgb_array'))
steps = 0
rr = 0
for ep in range(1):
    observation = env.reset()

    while True:
        _ = img.set_data(env.render(mode='rgb_array')) # just update the data
        display.display(plt.gcf())
        display.clear_output(wait=True)

        action = choose_action(observation)
        observation_, reward, done, _ = env.step(action)
        rr += reward

        store_transition(observation, action, reward, observation_)

        if steps > MIN_STEPS:
            learn()

        if done:
            print(f"Episode {ep}\nReward{reward}")
