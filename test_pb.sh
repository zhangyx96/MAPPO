source activate maddpg
CUDA_VISIBLE_DEVICES=1 python render_pb.py --env-name "push_ball_origin" --model_dir '/cur1_pb_244'  \
--adv_num 2 --good_num 4 --landmark_num 4
