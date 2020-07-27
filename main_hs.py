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
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#env = make_env("simple_spread", discrete_action=True)
#env.seed(1)

#torch.set_default_tensor_type('torch.DoubleTensor')
def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            #env.seed(seed + rank * 1000)
            #np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def main():
    args = get_args()
    model_dir = args.model_dir + '_' + str(args.adv_num) + str(args.good_num) + str(args.landmark_num)
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
    writer = SummaryWriter("/home/zhangyx/MAPPO/logs_2"+args.model_dir+"/validation")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                    args.gamma, args.log_dir, device, False)

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)
    # env_test = make_env("simple_spread", discrete_action=True)

    share_base = ATTBase(hidden_size=256)
    #share_base = ATTBase(envs.observation_space[0].shape[0], hidden_size=100)
    dist_movement = MultiCategorical(share_base.output_size, 11, 3) # movement的分布函数
    dist_pull = Categorical(share_base.output_size, 2) # pull的分布函数
    dist_lock = Categorical(share_base.output_size, 2) # lock的分布函数
    dists = [dist_movement, dist_pull, dist_lock]
    actor_critic_seeker = []
    actor_critic_hider = []
    for i in range(args.adv_num):
        ac = Policy(
            agent_i=i,
            base=share_base,
            dists=dists,
            base_kwargs={'recurrent': args.recurrent_policy})
        ac.to(device)
        actor_critic_seeker.append(ac)
    
    # for i in range(args.good_num):
    #     ac = Policy(
    #         agent_i=i,
    #         base=share_base,
    #         dists=dists,
    #         base_kwargs={'recurrent': args.recurrent_policy})
    #     ac.to(device)
    #     actor_critic_hider.append(ac)

    agent_seeker = []
    agent_hider = []
    for i in range(args.adv_num):
        agent_seeker.append(algo.PPO(
            actor_critic_seeker[i],
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
    # for i in range(args.good_num):
    #     agent_hider.append(algo.PPO(
    #         actor_critic_seeker[i],
    #         i,
    #         args.clip_param,
    #         args.ppo_epoch,
    #         args.num_mini_batch,
    #         args.value_loss_coef,
    #         args.entropy_coef,
    #         lr=args.lr,
    #         eps=args.eps,
    #         max_grad_norm=args.max_grad_norm,
    #         model_dir = args.model_dir))


     
    ## trainning configs
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    obs = envs.reset()
    observation_space_shape = len(obs[0][0])
    observation_action_shape = 5
    rollouts = []
    for i in range(args.adv_num):
        rollout = RolloutStorage(args.num_steps, args.num_processes,
                                observation_space_shape, observation_action_shape,
                                actor_critic_seeker[i].recurrent_hidden_state_size,
                                args.adv_num + args.good_num, i, args.assign_id)
        rollouts.append(rollout)
    for i in range(args.adv_num):
        rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
        rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
        rollouts[i].to(device)

    for j in range(num_updates):
        print("[%d/%d]"%(j,num_updates))
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            for i in range(args.adv_num):
                utils.update_linear_schedule(
                    agent_seeker[i].optimizer, j, num_updates,
                    agent_seeker[i].optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
            with torch.no_grad():
                for i in range(args.adv_num):
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic_seeker[i].act(
                        rollouts[i].share_obs[step],
                        rollouts[i].obs[step], args.adv_num, rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step])
                    value_list.append(value)
                    action_list.append(action)
                    action_log_prob_list.append(action_log_prob)
                    recurrent_hidden_states_list.append(recurrent_hidden_states)
                
            # Obser reward and next obs
            # action = []
            # for i in range(args.num_processes):
            #     one_env_action = []
            #     for k in range(args.adv_num):
            #         one_hot_action = np.zeros(envs.action_space[0].n)
            #         one_hot_action[action_list[k][i]] = 1
            #         one_env_action.append(one_hot_action)
            #     action.append(one_env_action)
            #start = time.time()
            #pdb.set_trace()

            action = []
            for i in range(args.num_processes):
                one_env_action = []
                for k in range(args.adv_num):
                    action_movement = MultiDiscrete(action_list[k][i][:3])
                    action_pull = Discrete(action_list[k][i][3])
                    action_lock = Discrete(action_list[k][i][4])
                    all_action = Tuple([action_movement, action_pull, action_lock])
                    one_env_action.append(all_action)
                action.append(one_env_action)
            import pdb; pdb.set_trace()


        
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            masks = torch.ones(args.num_processes, 1)
            bad_masks = torch.ones(args.num_processes, 1)
            #import pdb; pdb.set_trace()
            if args.assign_id:
                for i in range(args.adv_num):
                    vec_id = np.zeros((args.num_processes, args.adv_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
            else:
                for i in range(args.adv_num):
                    rollouts[i].insert(torch.tensor(obs.reshape(args.num_processes, -1)), torch.tensor(obs[:,i,:]), 
                                recurrent_hidden_states, action_list[i],
                                action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                    
        with torch.no_grad():
            next_value_list = []
            for i in range(args.adv_num):
                next_value = actor_critic[i].get_value(
                    rollouts[i].share_obs[-1],
                    rollouts[i].obs[-1], args.adv_num, rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1]).detach()
                next_value_list.append(next_value)

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])
        
        for i in range(args.adv_num):
            rollouts[i].compute_returns(next_value_list[i], args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

        for i in range(args.adv_num):
            value_loss, action_loss, dist_entropy = agent_seeker[i].update(rollouts[i], args.adv_num)
            #import pdb; pdb.set_trace()
            if (i == 0 and (j+1)%10 == 0):
                print("update num: " + str(j+1) + " value loss: " + str(value_loss))
        #rollouts.after_update()
        obs = envs.reset()
        if args.assign_id:
            for i in range(args.adv_num):    
                vec_id = np.zeros((args.num_processes, args.adv_num))
                vec_id[:, i] = 1
                vec_id = torch.tensor(vec_id)
                as_obs = torch.tensor(obs.reshape(args.num_processes, -1))
                a_obs = torch.tensor(obs[:,i,:])
                rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                rollouts[i].to(device)
        else:
            for i in range(args.adv_num):
                rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(args.num_processes, -1)))
                rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                rollouts[i].to(device)

   
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + model_dir):
                os.makedirs(save_path + model_dir)
            for i in range(args.adv_num):
                torch.save([
                    actor_critic[i],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], save_path + model_dir + '/agent_%i' % (i+1) + ".pt")

if __name__ == "__main__":
    main()