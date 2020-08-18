import copy
import glob
import os
import time
from collections import deque

import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model_poet_hs import Policy, ATTBase
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian, MultiCategorical
from a2c_ppo_acktr.storage_hs import RolloutStorage
from evaluation import evaluate

from utils.make_env_hs import make_env                         
from utils.env_wrappers_poet_hs import SubprocVecEnv, DummyVecEnv
from matplotlib import pyplot as plt 
import random
import copy
#import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os
from datetime import datetime
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def main():
    args = get_args()
    model_dir = args.model_dir + '_' + str(args.adv_num) + str(args.good_num) 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda" if args.cuda else "cpu")
    writer = SummaryWriter("/home/zhangyx/MAPPO/logs_2" + args.model_dir +  '_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)
    obs = envs.reset()

    observation_space_shape = len(obs[0][0])
    observation_action_shape = 5
    # env_test = make_env("simple_spread", discrete_action=True)
    share_base = ATTBase(input_size =observation_space_shape,  hidden_size=256)

    dist_movement = MultiCategorical(share_base.output_size, 11, 3) # movement的分布函数
    dist_pull = Categorical(share_base.output_size, 2) # pull的分布函数
    dist_lock = Categorical(share_base.output_size, 2) # lock的分布函数
    dists = [dist_movement, dist_pull, dist_lock]
    
    actor_critic_hider = []

    for i in range(args.good_num):
        ac = Policy(
            agent_i = i,
            base=share_base,
            dists=dists,
            base_kwargs={'recurrent': args.recurrent_policy})
        ac.to(device)
        actor_critic_hider.append(ac)
    
    agent_hider = []

    for i in range(args.good_num):
        agent_hider.append(algo.PPO(
            actor_critic_hider[i],
            i,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            model_dir = args.model_dir))
     
    ## trainning configs
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    rollouts = []
    for i in range(args.good_num):
        rollout = RolloutStorage(args.num_steps, args.num_processes,
                                observation_space_shape, observation_action_shape,
                                actor_critic_hider[i].recurrent_hidden_state_size,
                                args.adv_num + args.good_num, i, args.assign_id)
        rollouts.append(rollout)
        rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
        rollouts[i].obs[0].copy_(torch.tensor(obs[:, i, :]))
        rollouts[i].to(device)

    for j in range(num_updates):
        print("[%d/%d]"%(j,num_updates))
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(args.good_num):
                utils.update_linear_schedule(
                    agent_hider[i].optimizer, j, num_updates,
                    agent_hider[i].optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(args.num_steps):
            # Sample actions
            value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
            with torch.no_grad():
                for i in range(args.good_num):
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic_hider[i].act(
                        rollouts[i].share_obs[step],
                        rollouts[i].obs[step], args.good_num, rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step])
                    value_list.append(value)
                    action_list.append(action)
                    action_log_prob_list.append(action_log_prob)
                    recurrent_hidden_states_list.append(recurrent_hidden_states)
                
            #pdb.set_trace()
            action = []
            for i in range(args.num_processes):
                action_movement = []
                action_pull = []
                action_glueall = []
                action_movement.append(action_list[0][i][:3].cpu().numpy())
                action_pull.append(np.int(action_list[0][i][3].cpu().numpy()))
                action_glueall.append(np.int(action_list[0][i][4].cpu().numpy()))
                action_movement = np.stack(action_movement, axis = 0)
                action_pull = np.stack(action_pull, axis = 0)
                action_glueall = np.stack(action_glueall, axis = 0)
                one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
                #one_env_action = {'action_movement': action_movement}
                action.append(one_env_action)
            
            
            obs, reward, done, infos = envs.step(action, args.num_processes)
            reward = 
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            # bad_masks = torch.FloatTensor(
            #     [[0.0] if 'bad_transition' in info.keys() else [1.0]
            #      for info in infos[0]])

            # masks = torch.ones(args.num_processes, 1)
            bad_masks = torch.ones(args.num_processes, 1)
            if args.assign_id:
                for i in range(args.good_num):
                    vec_id = np.zeros((args.num_processes, args.good_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
            else:
                for i in range(args.good_num):
                    rollouts[i].insert(torch.tensor(obs.reshape(args.num_processes, -1)), torch.tensor(obs[:, i, :]), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)       
        #print('final_reward', reward_sum/args.num_steps)

        with torch.no_grad():
            next_value_list = []
            for i in range(args.good_num):
                next_value = actor_critic_hider[i].get_value(
                    rollouts[i].share_obs[-1],
                    rollouts[i].obs[-1], args.good_num, rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1]).detach()
                next_value_list.append(next_value)

        if args.gail:
            assert 0
        
        for i in range(args.good_num):
            rollouts[i].compute_returns(next_value_list[i], args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)
            value_loss, action_loss, dist_entropy = agent_hider[i].update(rollouts[i], args.good_num)
            
            if (i == 0 and (j+1)%10 == 0):
                print("update num:",str(j+1)," value loss: ", str(value_loss), "reward", str(rollouts[i].rewards.mean()))
        #rollouts.after_update()
        obs = envs.reset()
        if args.assign_id:
            for i in range(args.good_num):    
                vec_id = np.zeros((args.num_processes, args.good_num))
                vec_id[:, i] = 1
                vec_id = torch.tensor(vec_id)
                as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                a_obs = torch.tensor(obs[:,i,:])
                rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                rollouts[i].to(device)
        else:
            for i in range(args.good_num):
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:, i, :]))
                rollouts[i].to(device)

   
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + model_dir):
                os.makedirs(save_path + model_dir)
            for i in range(args.good_num):
                torch.save([
                    actor_critic_hider[i],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], save_path + model_dir + '/agent_%i' % (i+1) + ".pt")

if __name__ == "__main__":
    main()