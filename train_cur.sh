CUDA_VISIBLE_DEVICES=0 python main_cur.py --env-name "simple_spread" --model_dir '/run456_policyattention_mix' \
--save-interval 200 --algo ppo --use-gae --lr 2e-3 --clip-param 0.2 --value-loss-coef 1.2  \
--num-processes 2 --num-steps 80 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01  \
--agent_num 3 --ppo-epoch 25 --gae-lambda 0.98 --seed 1 --num-env-steps 500000000  \
--share_policy --use-linear-lr-decay --use_attention
