python main_dist_pong.py ps 0 &
CUDA_VISIBLE_DEVICES=0 python main_dist_pong.py worker 0 &
CUDA_VISIBLE_DEVICES=1 python main_dist_pong.py worker 1 &
CUDA_VISIBLE_DEVICES=2 python main_dist_pong.py worker 2 &
CUDA_VISIBLE_DEVICES=3 python main_dist_pong.py worker 3 &
CUDA_VISIBLE_DEVICES=4 python main_dist_pong.py worker 4 &
CUDA_VISIBLE_DEVICES=5 python main_dist_pong.py worker 5 &
CUDA_VISIBLE_DEVICES=6 python main_dist_pong.py worker 6 &
CUDA_VISIBLE_DEVICES=7 python main_dist_pong.py worker 7 &
