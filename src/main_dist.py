import numpy as np
import sys
import tensorflow as tf

from cartpole_agent import CartPoleAgent
from pong_agent import PongAgent

# Change this line to determine which game to play
Agent = CartPoleAgent

# The type of work and index is gathered from the args
# Ex. "python main_dist.py ps 0" starts parameter server 0
args = sys.argv
job_name = args[1]
task_index = int(args[2])

num_workers = 2

# Define the cluster spec with parameter servers and workers
# IP: are required manually, for larger scale, change to cluster manager
workers = ["localhost:222{}".format(i) for i in range(3, 3+num_workers)]
cluster_spec = tf.train.ClusterSpec({"ps": ["localhost:2222"], "worker": workers})

# Get the right server object depending on place in cluster
server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

# If parameter server, wait for instructions
if job_name == "ps":
    print("Starting server")
    server.join()

# Define our device based onn the replica device setter
# All workers are given same partition of tf graph
worker_device = "/job:worker/task:{}".format(task_index)
device = tf.train.replica_device_setter(cluster=cluster_spec, ps_device="/job:ps/cpu:0", worker_device=worker_device)
target = server.target

# The tf graph is defined in the Learner in the Agent
with tf.device(device):
    agent = Agent(task=task_index)

# Monitored Training Session makes sure the sessions on all workers are maintained properly
with tf.train.MonitoredTrainingSession(master=target,
                                       is_chief=(task_index == 0)) as sess:

    # The chief worker is in charge of initializing all parameters
    if task_index == 0:
        agent.init_params(sess)

    # Runs the interaction and training
    agent.run_episodes(sess)

    # After the training is done the chief stays idle
    if task_index == 0:
        server.join()
