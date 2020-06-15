import copy
import glob
import os
import time
from collections import deque

import gym
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
from a2c_ppo_acktr.model_poet_pb import Policy, ATTBase
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from utils.make_env_poet_pb import make_env                         
from utils.env_wrappers_poet_pb import SubprocVecEnv, DummyVecEnv
import pdb
import random
import copy
#import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#env = make_env("simple_spread", discrete_action=True)
#env.seed(1)

#torch.set_default_tensor_type('torch.DoubleTensor')

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

def SampleNearby(pos, max_step, TB, M):
    pos = pos + [] 
    pos_new = [] # save the new pos
    if pos == []:
        return []
    else:
        for i in range(len(pos)):
            pos_tmp = copy.deepcopy(pos[i])
            for j in range(TB):
                pos_tmp = pos_tmp + np.random.uniform(-max_step, max_step, 2) 
                pos_tmp = (pos_tmp+1)%2-1 #限制在(-1,+1)之间
                pos_new.append(copy.deepcopy(pos_tmp))
        return pos_new

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
    writer = SummaryWriter("/home/zhangyx/poet/logs"+args.model_dir+"/validation")

    #envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
    #                    args.gamma, args.log_dir, device, False)

    envs = make_parallel_env(args.env_name, args.num_processes, args.seed, True)
    # env_test = make_env("simple_spread", discrete_action=True)

    actor_critic = []
    if args.share_policy:
        if args.use_attention:
            share_base = ATTBase(envs.observation_space[0].shape[0], hidden_size=100)
            share_dist = Categorical(share_base.output_size, envs.action_space[0].n)
            for i in range(args.adv_num):
                ac = Policy(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_i=i,
                    agent_num=args.adv_num, 
                    adv_num=args.adv_num, 
                    good_num=args.good_num,
                    landmark_num=args.landmark_num,
                    base=share_base,
                    dist=share_dist,
                    base_kwargs={'recurrent': args.recurrent_policy})
                ac.to(device)
                actor_critic.append(ac)
        else:
            ac = Policy(
                    envs.observation_space[0].shape,
                    envs.action_space[0],
                    agent_num=args.adv_num, 
                    agent_i=0,
                    base_kwargs={'recurrent': args.recurrent_policy, 'assign_id': args.assign_id})
            ac.to(device)
            for i in range(args.adv_num):
                actor_critic.append(ac)
    else:
        for i in range(args.adv_num):
            ac = Policy(
                envs.observation_space[0].shape,
                envs.action_space[0],
                agent_num=args.adv_num, 
                agent_i=i,
                base_kwargs={'recurrent': args.recurrent_policy})
            ac.to(device)
            actor_critic.append(ac)
    #import pdb; pdb.set_trace()
    
    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        '''
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        '''
        agent = []
        for i in range(args.adv_num):
            agent.append(algo.PPO(
                actor_critic[i],
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
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)
    ##        
    ## trainning configs
    ##
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    now_agent_num = args.adv_num  #now_agent_num与adv_num一致
    #ratio = 1
    ## 给定初始的状态分布
    starts = []
    select_starts = []

    buffer_length = args.num_processes
    curri = 0
    N_new = 1000 # 每次扩展最多能放进archive的是多少个点
    max_step = 0.1 # sample_nearby 参数
    TB = 3  # sample_nearby 参数
    M = 5 * args.num_processes 
    Rmin = 0.2  # Reward阈值
    Rmax = 0.6  
    Cmin = 0.4  # cover_rate阈值
    Cmax = 0.8
    fix_iter = 10 # 保证经过fix_iter之后，再向外扩
    count_fix = 0 # 记录fix_training的轮数
    reproduce_flag = 0 # 是否生成新环境的标志
    eval_iter = 5  
    curri = 0

    ## 生成初始的环境
    pos_buffer = []
    starts_landmark = [] # save the landmarks pos in one env
    starts_balls = [] # save the balls pos in one env
    starts_agent = [] # save the agents pos in one env
    for j in range(args.num_processes): # sample the landmarks and balls
        for i in range(args.landmark_num):
            landmark_location = np.random.uniform(-1, +1, 2)   # landmark位置均匀分布
            ball_location = np.random.uniform(-0.2, +0.2, 2) + landmark_location # ball位置在landmark坐标周围均匀分布
            starts_landmark.append(copy.deepcopy(landmark_location)) # 存入list
            starts_balls.append(copy.deepcopy(ball_location))
        for i in range(args.adv_num): # sample the agents
            agent_location = np.random.uniform(-0.2, +0.2, 2) + ball_location # agent位置在最后一个ball坐标周围均匀分布
            starts_agent.append(copy.deepcopy(agent_location))
        pos_buffer.append(starts_agent + starts_balls + starts_landmark)
        starts_agent = []
        starts_balls = []
        starts_landmark = []


    for j in range(num_updates):
        print('[count_fix]:', count_fix)
        if not reproduce_flag: # 不生成新的环境，reproduce_flag初始为0
            cover_info_list = []
            now_num_processes_train = min(args.num_processes,len(pos_buffer)) # 并行环境数
            obs = envs.new_starts_obs(pos_buffer, now_agent_num, now_num_processes_train) # 获取初始obs
            ## 初始化rollouts
            rollouts = []
            for i in range(now_agent_num):
                rollout = RolloutStorage(args.num_steps, now_num_processes_train,
                                        envs.observation_space[0].shape, envs.action_space[0],
                                        actor_critic[i].recurrent_hidden_state_size,
                                        now_agent_num, i, args.assign_id)
                rollouts.append(rollout)
            ## 是否assign_id 默认False
            if args.assign_id:
                for i in range(now_agent_num):    
                    vec_id = np.zeros((now_num_processes_train, now_agent_num))
                    vec_id[:, i] = 1
                    vec_id = torch.tensor(vec_id)
                    as_obs = torch.tensor(obs.reshape(now_num_processes_train, -1))
                    a_obs = torch.tensor(obs[:,i,:])
                    rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                    rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                    rollouts[i].to(device)
            else:
                for i in range(now_agent_num):
                    rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(now_num_processes_train, -1)))
                    rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                    rollouts[i].to(device)
            ## 是否decrease learning rate linearly 默认False
            if args.use_linear_lr_decay:
                for i in range(now_agent_num):
                    utils.update_linear_schedule(
                        agent[i].optimizer, j, num_updates,
                        agent[i].optimizer.lr if args.algo == "acktr" else args.lr)
            ##
            ## start training
            ##
            for step in range(args.num_steps): 
                ## Sample actions
                value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
                with torch.no_grad():
                    for i in range(now_agent_num):
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                            rollouts[i].share_obs[step],
                            rollouts[i].obs[step], now_agent_num, rollouts[i].recurrent_hidden_states[step],
                            rollouts[i].masks[step])
                        value_list.append(value)
                        action_list.append(action)
                        action_log_prob_list.append(action_log_prob)
                        recurrent_hidden_states_list.append(recurrent_hidden_states)
                ## 对action进行one-hot编码
                action = []
                for i in range(now_num_processes_train):
                    one_env_action = []
                    for k in range(now_agent_num):
                        one_hot_action = np.zeros(envs.action_space[0].n)
                        one_hot_action[action_list[k][i]] = 1
                        one_env_action.append(one_hot_action)
                    action.append(one_env_action)
                
                obs, reward, done, infos = envs.step(action, now_num_processes_train)
                masks = torch.ones(now_num_processes_train, 1)
                bad_masks = torch.ones(now_num_processes_train, 1)

                if args.assign_id:
                    for i in range(now_agent_num):
                        vec_id = np.zeros((now_num_processes_train, now_agent_num))
                        vec_id[:, i] = 1
                        vec_id = torch.tensor(vec_id)
                        as_obs = torch.tensor(obs.reshape(now_num_processes_train, -1))
                        a_obs = torch.tensor(obs[:,i,:])
                        rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                    recurrent_hidden_states, action_list[i],
                                    action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                else:
                    for i in range(now_agent_num):
                        rollouts[i].insert(torch.tensor(obs.reshape(now_num_processes_train, -1)), torch.tensor(obs[:,i,:]), 
                                    recurrent_hidden_states, action_list[i],
                                    action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)

            writer.add_scalars('agent0/cover_rate',{'cover_rate': np.mean(infos)},j)
            for i in range(len(infos)):
                cover_info_list.append(infos[i][0]) 

            with torch.no_grad():
                next_value_list = []
                for i in range(now_agent_num):
                    next_value = actor_critic[i].get_value(
                        rollouts[i].share_obs[-1],
                        rollouts[i].obs[-1], now_agent_num, rollouts[i].recurrent_hidden_states[-1],
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
            
            for i in range(now_agent_num):
                rollouts[i].compute_returns(next_value_list[i], args.use_gae, args.gamma,
                                        args.gae_lambda, args.use_proper_time_limits)

            for i in range(now_agent_num):
                value_loss, action_loss, dist_entropy = agent[i].update(rollouts[i], now_agent_num)
                if (i == 0 and (j+1)%10 == 0):
                    print("update num: " + str(j+1) + " value loss: " + str(value_loss))

            ## 当满足fix_iter时，把最后一次满足parent条件的点取出
            print("cover_info_list: ", len(cover_info_list))
            if count_fix == fix_iter:
                count_fix = 0 # counter归零
                reproduce_flag = 1 # reproduce flag设为True
                parent_starts = [] # 保存parents_starts
                easy_count = 0 
                del_num = 0 # 删去的点数量
                for i in range(len(cover_info_list)):
                    if cover_info_list[i]> Cmax: # 如果cover_rate大于Cmax,放进parents_list
                        parent_starts.append(copy.deepcopy(pos_buffer[i-del_num]))  #index要减去删去的点的数量
                        del pos_buffer[i-del_num]
                        del_num += 1
                print('[parent_num]:', len(parent_starts))
            else:
                count_fix += 1
            print('##############################')

        ## reproduce_flag==1， 开始生成新的环境
        else:
            starts = copy.deepcopy(parent_starts) #此时的select_starts暂时定为，训练环节最后一个选出来的点(简单点)
            newsample_starts = SampleNearby(starts, max_step, TB, M)
            print("[newsample]:",len(newsample_starts))
            #需要测试逐个是否满足要求，然后把满足要求的加入到archive中
            curri = 0
            add_starts = []
            good_children_starts = []
            while curri < len(newsample_starts) and newsample_starts != []:
                if len(newsample_starts) - curri < args.num_processes:
                    starts = copy.deepcopy(newsample_starts[curri: len(newsample_starts)])
                    now_num_processes = len(newsample_starts)-curri
                    obs = envs.new_starts_obs(starts, now_agent_num, now_num_processes)
                    curri = len(newsample_starts)
                else:
                    starts = copy.deepcopy(newsample_starts[curri: curri + args.num_processes])
                    now_num_processes = args.num_processes
                    obs = envs.new_starts_obs(starts, now_agent_num, now_num_processes)
                    curri += args.num_processes
                rollouts = []
                for i in range(now_agent_num):
                    rollout = RolloutStorage(args.num_steps, now_num_processes,
                                            envs.observation_space[0].shape, envs.action_space[0],
                                            actor_critic[i].recurrent_hidden_state_size,
                                            now_agent_num, i, args.assign_id)
                    rollouts.append(rollout)
                # obs = envs.reset()
                if args.assign_id:
                    for i in range(now_agent_num):    
                        vec_id = np.zeros((now_num_processes, now_agent_num))
                        vec_id[:, i] = 1
                        vec_id = torch.tensor(vec_id)
                        as_obs = torch.tensor(obs.reshape(now_num_processes, -1))
                        a_obs = torch.tensor(obs[:,i,:])
                        rollouts[i].share_obs[0].copy_(torch.cat((as_obs, vec_id),1))
                        rollouts[i].obs[0].copy_(torch.cat((a_obs, vec_id),1))
                        rollouts[i].to(device)

                else:
                    for i in range(now_agent_num):
                        # pdb.set_trace()
                        rollouts[i].share_obs[0].copy_(torch.tensor(obs.reshape(now_num_processes, -1)))
                        rollouts[i].obs[0].copy_(torch.tensor(obs[:,i,:]))
                        rollouts[i].to(device)

                if args.use_linear_lr_decay:
                    # decrease learning rate linearly
                    for i in range(now_agent_num):
                        utils.update_linear_schedule(
                            agent[i].optimizer, j, num_updates,
                            agent[i].optimizer.lr if args.algo == "acktr" else args.lr)

                for step in range(args.num_steps):
                    # Sample actions
                    value_list, action_list, action_log_prob_list, recurrent_hidden_states_list = [], [], [], []
                    with torch.no_grad():
                        for i in range(now_agent_num):
                            #pdb.set_trace()
                            value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                                rollouts[i].share_obs[step],
                                rollouts[i].obs[step], now_agent_num, rollouts[i].recurrent_hidden_states[step],
                                rollouts[i].masks[step])
                            value_list.append(value)
                            action_list.append(action)
                            action_log_prob_list.append(action_log_prob)
                            recurrent_hidden_states_list.append(recurrent_hidden_states)
                    # Obser reward and next obs
                    action = []
                    for i in range(now_num_processes):
                        one_env_action = []
                        for k in range(now_agent_num):
                            one_hot_action = np.zeros(envs.action_space[0].n)
                            one_hot_action[action_list[k][i]] = 1
                            one_env_action.append(one_hot_action)
                        action.append(one_env_action)
                    obs, reward, done, infos = envs.step(action, now_num_processes)
                    masks = torch.ones(now_num_processes, 1)
                    bad_masks = torch.ones(now_num_processes, 1)

                    if args.assign_id:
                        for i in range(now_agent_num):
                            vec_id = np.zeros((now_num_processes, now_agent_num))
                            vec_id[:, i] = 1
                            vec_id = torch.tensor(vec_id)
                            as_obs = torch.tensor(obs.reshape(now_num_processes, -1))
                            a_obs = torch.tensor(obs[:,i,:])
                            rollouts[i].insert(torch.cat((as_obs, vec_id),1), torch.cat((a_obs, vec_id),1), 
                                        recurrent_hidden_states, action_list[i],
                                        action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                    else:
                        for i in range(now_agent_num):
                            rollouts[i].insert(torch.tensor(obs.reshape(now_num_processes, -1)), torch.tensor(obs[:,i,:]), 
                                        recurrent_hidden_states, action_list[i],
                                        action_log_prob_list[i], value_list[i], torch.tensor(reward[:, i].reshape(-1,1)), masks, bad_masks)
                for i in range(len(infos)):# 这里需要换成后五帧的信息
                    if infos[i][0] < Cmax and infos[i][0] > Cmin:
                        good_children_starts.append(copy.deepcopy(starts[i]))
            if len(good_children_starts) > N_new:
                add_starts = random.sample(good_children_starts, N_new)
            else:
                add_starts = copy.deepcopy(good_children_starts)
            print("add_num: ",len(add_starts))
            if len(pos_buffer) + len(add_starts) <= buffer_length:
                pos_buffer = pos_buffer + add_starts
                curri = len(pos_buffer)-1
            else:
                pos_buffer = pos_buffer + add_starts
                pos_buffer = pos_buffer[len(pos_buffer)-buffer_length:]
            print('old_length: ', len(pos_buffer))
            reproduce_flag = 0

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            if not os.path.exists(save_path + args.model_dir):
                os.makedirs(save_path + args.model_dir)
            # 存储agent数最大值的参数
            for i in range(now_agent_num):
                torch.save([
                    actor_critic[i],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], save_path + args.model_dir + '/agent_%i' % (i+1) + ".pt")


if __name__ == "__main__":
    main()