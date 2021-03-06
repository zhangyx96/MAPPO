ulimit -n 16384
python main_poet_sp.py --env-name "simple_spread" --model_dir '/sp_poet' --save-interval 200 --algo ppo --use-gae --lr 2e-3 --clip-param 0.2 --value-loss-coef 0.5 \
--num-processes 2000 --num-steps 70 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 \
--agent_num 4 --ppo-epoch 25 --gae-lambda 0.98 --seed 1 --num-env-steps 600000000 --share_policy --use-linear-lr-decay --use_attention 
