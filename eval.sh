source activate mappo
CUDA_VISIBLE_DEVICES=1 python PPO_evaluate.py --model_dir '/run4_reverse_true_0.5' --agent_num 4
