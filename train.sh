ulimit -n 4096 
CUDA_VISIBLE_DEVICES=0 python main.py --env-name "simple_tag_coop" --model_dir '/pp_test' \
--save-interval 200 --algo ppo --use-gae --lr 2e-3 --clip-param 0.2 --value-loss-coef 0.5  \
--num-processes 500 --num-steps 80 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 \
--agent_num 4 --ppo-epoch 25 --gae-lambda 0.98 --seed 1 --num-env-steps 300000000 \
--share_policy --use-linear-lr-decay --use_attention
