import numpy as np
import sys
import tensorflow as tf

from regression_agent import RegressionAgent

args = sys.argv

job_name = args[1]
task_index = int(args[2])

cluster_spec = tf.train.ClusterSpec({"ps": ["localhost:2220"], "worker": ["localhost:2230", "localhost:2231", "localhost:2232"]})

server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

if job_name == "ps":
    print("Starting server")
    server.join()

worker_device = "/job:worker/task:{}".format(task_index)
device = tf.train.replica_device_setter(cluster=cluster_spec, worker_device=worker_device)
target = server.target

with tf.device(device):
    agent = RegressionAgent()


hooks = [tf.train.StopAtStepHook(last_step=2000)]

with tf.train.MonitoredTrainingSession(master=target,
                                       is_chief=(task_index == 0),
                                       hooks=hooks) as sess:
    if task_index == 0:
        agent.init_params(sess)

    while not sess.should_stop():
        agent.play_and_train_epoch(sess)

    if task_index == 0:
        server.join()


