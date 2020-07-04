source activate maddpg
CUDA_VISIBLE_DEVICES=1 python render_pb.py --env-name "push_ball_origin" --model_dir '/cur1_pb'  \
--adv_num 2 --good_num 2 --landmark_num 2
