#!/bin/bash

BASE_LOG_DIR="/tmp"
TRAIN_LOG_DIR="$BASE_LOG_DIR/train"

NUM_PARAM_SERVERS=1
PARAM_SERVER_PORT=2220
WORKER_PORT=2230

num_gpus=$3
NUM_WORKERS=2

ps_hosts=""
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  port=$(($PARAM_SERVER_PORT + $i))
  if [ -z $ps_hosts ]; then
    ps_hosts="localhost:$port"
  else
    ps_hosts="$ps_hosts,localhost:$port"
  fi

  i=$(($i + 1))
done

worker_hosts=""
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  port=$(($WORKER_PORT + $i))
  if [ -z $worker_hosts ]; then
    worker_hosts="localhost:$port"
  else
    worker_hosts="$worker_hosts,localhost:$port"
  fi

  i=$(($i + 1))
done

echo $ps_hosts
echo $worker_hosts

# Start the param servers
i=0
while [[ $i -lt $NUM_PARAM_SERVERS ]]
do
  outfile="$BASE_LOG_DIR/ps$i"
  echo "Starting param server $i. Stdout: $outfile, train logs: $TRAIN_LOG_DIR."

  python distributed_test_regression.py
  --logdir=$TRAIN_LOG_DIR \
  --ps_hosts=$ps_hosts \
  --worker_hosts=$worker_hosts \
  --job="ps" \
  --task_id=$i \
  > $outfile 2>&1 &

  i=$(($i + 1))
done

# Start the worker instances for each GPU
i=0
while [[ $i -lt $NUM_WORKERS ]]
do
  outfile="$BASE_LOG_DIR/worker$i"
  gym_log="$GYM_LOG_DIR$i"
  log_params="--logdir=$TRAIN_LOG_DIR"

  echo "Starting worker $i. Stdout: $outfile, train logs: $TRAIN_LOG_DIR, "
  python distributed_test_regression.py
  --logdir=$TRAIN_LOG_DIR \
  --ps_hosts=$ps_hosts \
  --worker_hosts=$worker_hosts \
  --job="worker" \
  --task_id=$i \
  --gpu_id=$(($i + 1)) \
  > $outfile 2>&1 &

  i=$(($i + 1))
done
