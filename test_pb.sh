source activate maddpg
CUDA_VISIBLE_DEVICES=1 python render_pb.py --env-name "push_ball_poet" --model_dir '/pb_poet'  \
--adv_num 2 --good_num 1 --landmark_num 1
