import tensorflow as tf
import sys
import numpy as np

args = sys.argv

job_name = args[1]
task_index = int(args[2])

cluster_spec = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker": ["localhost:2223", "localhost:2224"]})

server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)


if job_name == "ps":
    print("Starting server")
    server.join()
print(1)


worker_device = "/job:worker/task:{}".format(task_index)
device = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=worker_device)
target = server.target

with tf.device(device):
    k = tf.Variable(0.0, dtype=tf.float32)
    m = tf.Variable(0.0, dtype=tf.float32)

    x = tf.placeholder(dtype=tf.float32, shape=[None])
    y = tf.scalar_mul(k, x) + m

    y_hat = tf.placeholder(dtype=tf.float32, shape=[None])

    loss = tf.losses.mean_squared_error(y_hat, y)

    optimizer = tf.train.AdamOptimizer(0.01)
    train_op = optimizer.minimize(loss=loss)

    init_op = tf.global_variables_initializer()

    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
    step_op = global_step_tensor.assign_add(1)


hooks = [tf.train.StopAtStepHook(last_step=500)]

with tf.train.MonitoredTrainingSession(master=target,
                                       is_chief=(task_index == 0),
                                       hooks=hooks) as sess:
    if task_index == 0:
        sess.run(init_op)

    while not sess.should_stop():
        K = 3
        M = -5
        X = np.random.normal(0, 10, 100)
        epsilon = np.random.normal(0, 0.01)
        Y = K * X + M + epsilon

        feed_dict = {x: X, y_hat: Y}
        _, p1, p2, _ = sess.run([train_op, k, m, step_op], feed_dict=feed_dict)
        print([p1, p2])
        #sess.run(step_op)

    if task_index == 0:
        server.join()


