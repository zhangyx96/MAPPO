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
save_gifs = False
seed = 1
log_interval = 10

def num_reach(world):
    num = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        if min(dists) <= world.agents[0].size + world.landmarks[0].size:
            num = num + 1
    return num 

if __name__ == '__main__':
    args = get_args()
    env = make_env(args.env_name, discrete_action=True)
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].n
    #torch.manual_seed(seed)
    #env.seed(seed)
    Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])
    n_episodes = 100
    episode_length = 80
    num_agent = args.agent_num
    ifi = 1 / 30
    gif_path = './gifs'
    success_rate = 0
    num_success = 0
    model_dir = args.model_dir

    agents = []
    for i in range(num_agent):
        # actor_critic, ob_rms = torch.load('/home/chenjy/new_version/trained_models/ppo' + args.model_dir + '/agent_%i' % (i+1) + ".pt")
        actor_critic, ob_rms = torch.load('/home/zhangyx/MAPPO/trained_models/ppo' + args.model_dir + '/agent_1' + ".pt")
        agents.append(actor_critic)
    if not os.path.exists('./gifs/' + model_dir):
        os.makedirs('./gifs/' + model_dir)


    recurrent_hidden_states = torch.zeros(1,agents[0].recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    cover_rate_sum = 0
    for ep_i in range(n_episodes):
        print("Episode %i of %i" % (ep_i + 1, n_episodes))
        obs = env.reset()
        if save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])
        for t_i in range(episode_length):
            calc_start = time.time()
            state = torch.tensor([state for state in obs], dtype=torch.float).cuda()
            actions = []
            al_agent = []
            al_landmark = []
            share_obs = state.view(1,-1)
            for i in range(num_agent):
                obs = state[i].view(-1, num_state)
                # value, action, _, recurrent_hidden_states, alpha_agent, alpha_landmark = agents[i].act(share_obs, obs, nagents, i, recurrent_hidden_states, masks)
                value, action, _, recurrent_hidden_states = agents[i].act(share_obs, obs, num_agent, i, recurrent_hidden_states, masks)
                actions.append(action)
                # al_agent.append(alpha_agent)
                # al_landmark.append(alpha_landmark)
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

            #import pdb; pdb.set_trace()
            # start = time.time()
            obs, rewards, dones, infos = env.step(actions)
            #pdb.set_trace()
            # end = time.time()
            # print('env', round(end-start,4))
            masks.fill_(0.0 if dones else 1.0)
            #import pdb; pdb.set_trace()
            # start = time.time()
            if save_gifs:
                frames.append(env.render('rgb_array')[0])
            # end = time.time()
            # print('render one', round(end-start,4))
            # start = time.time()
            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            # env.render('human')
            # end = time.time()
            # print('sleep', round(end-start,4))

            if t_i == episode_length -1:
                reach_num = num_reach(env.world)
                # writer.add_scalar('number_reach' , num_reach, ep_i)
                # writer.add_scalar('cover rate' , num_reach/nagents, ep_i)
                print('number_reach', reach_num)
                print('cover rate once', reach_num/num_agent)
                cover_rate_sum = cover_rate_sum + reach_num/num_agent

        if save_gifs:
            gif_num = 0
            while os.path.exists('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i)):
                gif_num += 1
            imageio.mimsave('./gifs/' + model_dir + '/%i_%i.gif' % (gif_num, ep_i),
                            frames, duration=ifi)
    print('cover_rate', cover_rate_sum/n_episodes)
    # print('success rate',num_success/n_episodes)
