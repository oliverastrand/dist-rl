import numpy as np
import sys
import tensorflow as tf

from cartpole_agent import Agent

args = sys.argv

job_name = args[1]
task_index = int(args[2])

workers = ["localhost:222{}".format(i) for i in range(3, 3+8)]
cluster_spec = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker": workers})

server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

if job_name == "ps":
    print("Starting server")
    server.join()

worker_device = "/job:worker/task:{}".format(task_index)
device = tf.train.replica_device_setter(cluster=cluster_spec, ps_device="/job:ps/cpu:0", worker_device=worker_device)
target = server.target

with tf.device(device):
    agent = Agent(task=task_index)

hooks = [tf.train.StopAtStepHook(last_step=2000)]

with tf.train.MonitoredTrainingSession(master=target,
                                       is_chief=(task_index == 0)) as sess: #,
                                       #hooks=hooks) as sess:
    if task_index == 0:
        agent.init_params(sess)

    #while not sess.should_stop():
    #    agent.play_and_train_epoch(sess)

    agent.run_episodes(sess)

    if task_index == 0:
        server.join()


