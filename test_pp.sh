source activate mappo
CUDA_VISIBLE_DEVICES=1 python render_pp.py --env-name "simple_tag_coop_rb" --model_dir '/pp_reborn_641'  \
--adv_num 6 --good_num 4 --landmark_num 1
