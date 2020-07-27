"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # import time; start = time.time()
            ob, reward, done, info = env.step(data)
            # end = time.time()
            # if end-start > 0.7:
            #     print('********')
            # print('one_step: ',end-start)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd[0] == 'new_starts_obs':
            now_agent_num = cmd[1]
            starts = cmd[2]
            index_index = cmd[3]
            ob = env.new_starts_obs(starts,now_agent_num,index_index)
            remote.send(ob)
        elif cmd[0:5] == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd[0:8] == "init_set":
            now_agent_num = int(cmd[8:])
            ob = env.init_set(now_agent_num)
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        #self.remotes[0].send(('get_agent_types', None))
        # self.agent_types = self.remotes[0].recv()
        self.length = len(env_fns)
        VecEnv.__init__(self, self.length, observation_space, action_space)

    def step_async(self, actions, now_num_processes):
        # import time; 
        # start = time.time()
        i = 0
        for remote, action in zip(self.remotes, actions):
            # import pdb; pdb.set_trace()
            if i < now_num_processes:
                remote.send(('step', action))
                i += 1
        # end = time.time()
        # print('time_send: ',end-start)
        self.waiting = True

    def step_wait(self, now_num_processes):
        # import time; start = time.time()
        results = []
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        # results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        # end = time.time()
        # print('time_get: ',end-start)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    # def reset(self):
    #     for remote in self.remotes:
    #         remote.send(('reset', None))
    #     return np.stack([remote.recv() for remote in self.remotes])

    def new_starts_obs(self, starts, now_agent_num, now_num_processes):
        tmp_list = ['new_starts_obs', now_agent_num, starts]
        i = 0
        results = []
        for remote in self.remotes:
            if i < now_num_processes:
                index_index = [i]
                remote.send((tmp_list + index_index, None))
                i += 1
        i = 0
        for remote in self.remotes:
            if i < now_num_processes:
                results.append(remote.recv())
                i += 1
        return np.stack(results)

    # using by main_reverse 
    # def new_starts_obs(self, starts, now_agent_num):
    #     self.new_starts_obs_async(starts, sample_index, now_agent_num)
    #     return self.new_starts_obs_wait()

    # def new_starts_obs_async(self, starts, sample_index, now_agent_num):
    #     tmp_list = ['new_starts_obs', now_agent_num, starts, sample_index]
    #     i = 0
    #     for remote in self.remotes:
    #         index_index = [i] # sample_index的index
    #         remote.send((tmp_list + index_index, None))
    #         i += 1
    #     self.waiting = True

    # def new_starts_obs_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     self.waiting = False
    #     obs, rou_index = zip(*results)
    #     return np.stack(obs), np.stack(rou_index)

    def init_set(self, now_agent_num, ratio):
        hard_num = int(self.length * ratio)
        count = 0
        for remote in self.remotes:
            if count <= hard_num:
                remote.send(('init_set' +  str(now_agent_num), None))
            else:
                remote.send(('reset' + str(now_agent_num), None))
            count = count + 1
        return np.stack([remote.recv() for remote in self.remotes])

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        tmp = np.stack([remote.recv() for remote in self.remotes])
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, self.length, observation_space, action_space)
        return tmp

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]

class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.length = len(env_fns) 
        VecEnv.__init__(self, self.length, env.observation_space, env.action_space)
        # if all([hasattr(a, 'adversary') for a in env.agents]):
        #     self.agent_types = ['adversary' if a.adversary else 'agent' for a in
        #                         env.agents]
        # else:
        #     self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos
    
    # def new_starts_obs(self, starts, now_agent_num):
    #     seed = 1
    #     obs,rou_index = env.new_starts_obs(starts,now_agent_num,seed)
    #     return np.array(obs)

    def init_set(self, now_agent_num):
        results = [env.init_set(now_agent_num) for env in self.envs]
        return np.array(results)
    # def reset(self):        
    #     results = [env.reset() for env in self.envs]
    #     return np.array(results)

    def reset(self):        
        results = [env.reset() for env in self.envs]
        env = self.envs[0]
        VecEnv.__init__(self, self.length, env.observation_space, env.action_space)
        # import pdb; pdb.set_trace()  
        return np.array(results)

    def close(self):
        return