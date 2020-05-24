import argparse
import pickle
from collections import namedtuple
from itertools import count

import os, time
import numpy as np
#import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from torch.autograd import Variable
import imageio
#from a2c_ppo_acktr.model import MLPBase, CNNBase
from a2c_ppo_acktr.arguments import get_args
import pdb
import time

# Parameters
gamma = 0.95
save_gifs = True
seed = 1
log_interval = 10
if __name__ == '__main__':
    args = get_args()
    n_episodes = 50
    episode_length = 100
    ifi = 1 / 30
    gif_path = './gifs'
    success_rate = 0
    num_success = 0
    model_dir = args.model_dir
    env = make_env(args.env_name, discrete_action=True)
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].n
    agents = []
    for i in range(args.adv_num):
        actor_critic, ob_rms = torch.load('/home/zhangyx/MAPPO/trained_models/ppo' + args.model_dir + '/agent_1' + ".pt")
        actor_critic.update_num(args.adv_num, args.good_num, args.landmark_num)
        agents.append(actor_critic)
    if not os.path.exists('./gifs/' + model_dir):
        os.makedirs('./gifs/' + model_dir)

    recurrent_hidden_states = torch.zeros(1,agents[0].recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    cover_rate_sum = 0
    for ep_i in range(n_episodes):
        print("Episode %i of %i" % (ep_i + 1, n_episodes))
        mean_reward = 0
        obs = env.reset()
        # obs = env.init_set(nagents)
        if save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        # env.render('human')
        for t_i in range(episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            state = torch.tensor([state for state in obs], dtype=torch.float).cuda()
            # get actions as torch Variables
            #import pdb; pdb.set_trace()
            ### 手动拼agent的state
            actions = []
            al_agent = []
            al_landmark = []
            share_obs = state.view(1,-1)
            for i in range(args.adv_num):
                obs = state[i].view(-1, num_state)
                value, action, _, recurrent_hidden_states = agents[i].act(share_obs, obs, args.adv_num, i, recurrent_hidden_states, masks)
                actions.append(action)
            torch_actions = actions

            # convert actions to numpy arrays
            prob_actions = [ac.data.cpu().numpy().flatten() for ac in torch_actions]
            #import pdb; pdb.set_trace()
            ### 手动生成one-hot向量
            actions = []
            for a in prob_actions:
                #index = np.argmax(a)
                ac = np.zeros(num_action)
                ac[a] = 1
                actions.append(ac)
            obs, rewards, dones, infos = env.step(actions)
            print(rewards[0])
            masks.fill_(0.0 if dones else 1.0)
            if save_gifs:
                frames.append(env.render('rgb_array')[0])
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
        if save_gifs:
            gif_num = 0
            while os.path.exists('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i)):
                gif_num += 1
            imageio.mimsave('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i),
                            frames, duration=ifi)

