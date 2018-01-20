import numpy as np
import sys
import tensorflow as tf

from regression_agent import RegressionAgent

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Distribution
    parser.add_argument('--ps_hosts',
    help='Parameter Servers. Comma separated list of host:port pairs')
    parser.add_argument('--worker_hosts',
    help='Worker hosts. Comma separated list of host:port pairs')
    parser.add_argument('--job', choices=['ps', 'worker'],
    help='Whether this instance is a param server or a worker')
    parser.add_argument('--task_id', type=int,
    help='Index of this task within the job')
    parser.add_argument('--gpu_id', type=int,
    help='Index of the GPU to run the training on')

    # Summary
    parser.add_argument('--logdir', default='/tmp/train_logs',
    help='Directory for training summary and logs')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    ps_hosts = args.ps_hosts.split(',') if args.ps_hosts else []
    worker_hosts = args.worker_hosts.split(',') if args.worker_hosts else []
    print(args.ps_hosts, args.worker_hosts)
    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    server = tf.train.Server(cluster_spec, job_name=args.job, task_index=args.task_id)

    task_index = args.task_id

    if args.job == 'ps':
        print("Starting server")
        server.join()

    worker_device = "/job:worker/task:{}".format(args.task_id)
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
