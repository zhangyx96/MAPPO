ulimit -n 16384
CUDA_VISIBLE_DEVICES=0 python main_bl.py --env-name "box_locking" --model_dir '/box_locking' --save-interval 500 --algo ppo --use-gae --lr 3e-3 --clip-param 0.2 --value-loss-coef 0.5 \
--num-processes 300 --num-steps 240 --num-mini-batch 1 --log-interval 1 --entropy-coef 0.01 \
--adv_num 0 --good_num 1 --ppo-epoch 25 --gae-lambda 0.98 --seed 1 --num-env-steps 600000000 --share_policy --use-linear-lr-decay --use_attention 
