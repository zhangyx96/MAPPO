source activate mappo
CUDA_VISIBLE_DEVICES=1 python render_sp.py --env-name "simple_spread_origin" --model_dir '/simple_spread_cur_4'  \
--agent_num 4
