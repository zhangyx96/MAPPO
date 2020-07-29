import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
import time

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, agent_i, dists, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
                self.base = base(obs_shape[0], agent_num, **base_kwargs)
            elif len(obs_shape) == 1:
                base = MLPBase
                #base = ATTBase
                self.base = base(obs_shape[0], agent_num, **base_kwargs)
            else:
                raise NotImplementedError
        else:
            self.base = base
        # self.base = base(obs_shape[0], **base_kwargs)
        # actor输入维度num_state，critic输入num_state*agent_num

        # self.base = base(obs_shape[0], agent_num, agent_i, **base_kwargs)

        #import pdb; pdb.set_trace()
        self.agent_i = agent_i
        self.dists = dists
        self.dists = nn.ModuleList(self.dists)
        

        

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, share_inputs, inputs, agent_num, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(share_inputs, inputs, self.agent_i, rnn_hxs, masks)
        
        dists = [dist(actor_features) for dist in self.dists]
        

    
        if deterministic:
            actions = [dist.mode() for dist in dists]
        else:
            #actions = [dist.sample() for dist in dists]
            actions = []
            for dist in dists:
                if isinstance(dist, list):
                    for fc in dist:    # fc refers FixedCategorical class
                        actions.append(fc.sample())
                else:
                    actions.append(dist.sample())

        action_log_probs = []
        action_count = 0
        for dist in dists:
            if isinstance(dist, list):
                for fc in dist:
                    action_log_probs.append(fc.log_probs(actions[action_count]))
                    action_count += 1
            else:
                action_log_probs.append(dist.log_probs(actions[action_count]))
                action_count += 1
        action_log_probs = torch.stack(action_log_probs, dim = 2).squeeze(1)
        actions = torch.stack(actions, dim = 2).squeeze(1)
        
        #action_log_probs = [dist.log_probs(action) for dist, action in zip(dists, actions)]
        # dist_entropy = [dist.entropy().mean() for dist in dists]
        return value, actions, action_log_probs, rnn_hxs
    
    # def update_num(self,adv_num, good_num, landmark_num):
    #     self.adv_num = adv_num
    #     self.good_num = good_num
    #     self.landmark_num = landmark_num


    def get_value(self, share_inputs, inputs, agent_num, rnn_hxs, masks):
        value, _, _ = self.base(share_inputs, inputs, self.agent_i, rnn_hxs, masks)
        return value

    def evaluate_actions(self, share_inputs, inputs, agent_num, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(share_inputs, inputs, self.agent_i, rnn_hxs, masks)

        dists = [dist(actor_features) for dist in self.dists]



        #actions = [dist.sample() for dist in dists]
        actions = []
        dist_entropy = []
        for dist in dists:
            if isinstance(dist, list):
                for fc in dist:    # fc refers FixedCategorical class
                    actions.append(fc.sample())
                    dist_entropy.append(fc.entropy().mean())
            else:
                actions.append(dist.sample())
                dist_entropy.append(dist.entropy().mean())
        dist_entropy = torch.stack(dist_entropy).mean()
                    

        action_log_probs = []
        action_count = 0
        for dist in dists:
            if isinstance(dist, list):
                for fc in dist:
                    action_log_probs.append(fc.log_probs(actions[action_count]))
                    action_count += 1
            else:
                action_log_probs.append(dist.log_probs(actions[action_count]))
                action_count += 1
        action_log_probs = torch.stack(action_log_probs, dim = 2).squeeze(1)
                
        return value, action_log_probs, dist_entropy, rnn_hxs

    


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, agent_num, recurrent=False, assign_id=False, hidden_size=100):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if assign_id:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs + agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs * agent_num + agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs * agent_num, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, share_inputs, inputs, agent_i, rnn_hxs, masks):
        #import pdb; pdb.set_trace()
        share_obs = share_inputs
        obs = inputs

        #if self.is_recurrent:
        #    x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(share_inputs)
        hidden_actor = self.actor(inputs)
        
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


# hide and seek        
class ObsEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super(ObsEncoder, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.self_encoder = nn.Sequential(
                            init_(nn.Linear(10, hidden_size)), nn.Tanh())
        self.seeker_encoder = nn.Sequential(
                            init_(nn.Linear(10, hidden_size)), nn.Tanh())
        self.hider_encoder = nn.Sequential(
                            init_(nn.Linear(10, hidden_size)), nn.Tanh())                            
        self.box_encoder = nn.Sequential(
                            init_(nn.Linear(12, hidden_size)), nn.Tanh())
        self.ramp_encoder = nn.Sequential(
                            init_(nn.Linear(12, hidden_size)), nn.Tanh())

        self.adv_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.adv_correlation_mat.data, gain=1)

        self.good_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.good_correlation_mat.data, gain=1)

        self.box_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.box_correlation_mat.data, gain=1)

        self.ramp_correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.ramp_correlation_mat.data, gain=1)

        self.fc = nn.Sequential(
                    init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        #self.encoder_linear = init_(nn.Linear(3*hidden_size, hidden_size))
        # 加上激活函数 效果会有比较大的提升 虽然还是达不到标准
        self.encoder_linear = nn.Sequential(
                            init_(nn.Linear(hidden_size * 5, hidden_size)), nn.Tanh(),
                            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


    def forward(self, inputs, adv_num, good_num, box_num, ramp_num):
        # 1 ~ 10 self
        # 11 ~ 10+n*10 other
        # 11+n*10 ~ 10+n*10+m*12  box
        # 
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        self_inputs = inputs[:, :10]
        other_nums = adv_num + good_num -1 # 除了自身以外其余agent得数量
        other_seeker_inputes = inputs[:, 10 : 10 + 10*(adv_num - 1)]
        hider_inputs = inputs[:, 10 + 10*(adv_num - 1) : 10 + 10*other_nums]
        box_inputs = inputs[:, 10 + 10 * other_nums : 10 + 10 * other_nums + 13 * box_num] #13 包含了mask_ab_obs，在最后
        ramp_inputs = inputs[:, 10 + 10 * other_nums + 13 * box_num : 10 + 10 * other_nums + 13 * box_num + 12 * ramp_num]

        emb_self = self.self_encoder(self_inputs)
        
        emb_seeker = []
        beta_seeker = []
        emb_hider = []
        beta_hider = []
        emb_box = []
        beta_box = []
        emb_ramp = []
        beta_ramp = []

        beta_seeker_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.adv_correlation_mat)
        beta_hider_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.good_correlation_mat)
        beta_box_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.box_correlation_mat)
        beta_ramp_ij = torch.matmul(emb_self.view(batch_size,1,-1), self.ramp_correlation_mat)
        
        for i in range(adv_num-1):
            emb_seeker.append(other_seeker_inputes[:, 10*i:10*(i+1)])

        for i in range(good_num):
            emb_hider.append(hider_inputs[:, 10*i:10*(i+1)])
        
        for i in range(box_num):
            emb_box.append(box_inputs[:, 12*i:12*(i+1)])
        
        for i in range(ramp_num):
            emb_ramp.append(ramp_inputs[:, 12*i:12*(i+1)])

        emb_seeker = torch.stack(emb_seeker,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_seeker = self.seeker_encoder(emb_seeker)
        beta_seeker = torch.matmul(beta_seeker_ij, emb_seeker.permute(0,2,1)).squeeze(1)

        
        emb_hider = torch.stack(emb_hider,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_hider = self.hider_encoder(emb_hider)
        beta_hider = torch.matmul(beta_hider_ij, emb_hider.permute(0,2,1)).squeeze(1)

        emb_box = torch.stack(emb_box,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_box = self.box_encoder(emb_box)
        beta_box = torch.matmul(beta_box_ij, emb_box.permute(0,2,1)).squeeze(1)

        emb_ramp = torch.stack(emb_ramp,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        emb_ramp = self.ramp_encoder(emb_ramp)
        beta_ramp = torch.matmul(beta_ramp_ij, emb_ramp.permute(0,2,1)).squeeze(1)

        alpha_seeker = F.softmax(beta_seeker,dim = 1).unsqueeze(2)   
        alpha_hider = F.softmax(beta_hider,dim = 1).unsqueeze(2)   
        alpha_box = F.softmax(beta_box,dim = 1).unsqueeze(2)
        alpha_ramp = F.softmax(beta_ramp,dim = 1).unsqueeze(2)

        vi_seeker = torch.mul(alpha_seeker,emb_seeker)
        vi_seeker = torch.sum(vi_seeker,dim=1)
        vi_hider = torch.mul(alpha_hider,emb_hider)
        vi_hider = torch.sum(vi_hider,dim=1)
        vi_box = torch.mul(alpha_box,emb_box)
        vi_box = torch.sum(vi_box,dim=1)
        vi_ramp = torch.mul(alpha_ramp,emb_ramp)
        vi_ramp = torch.sum(vi_ramp,dim=1)

        gi = self.fc(emb_self)
        f = self.encoder_linear(torch.cat([gi, vi_seeker, vi_hider, vi_box, vi_ramp], dim=1))
        return f



class ATTBase(NNBase):
    def __init__(self, num_inputs = 0, recurrent=False, assign_id=False, hidden_size=100):
        super(ATTBase, self).__init__(recurrent, num_inputs, hidden_size)
        if recurrent:
            num_inputs = hidden_size
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))    
        self.actor = ObsEncoder(hidden_size=hidden_size)
        self.encoder = ObsEncoder(hidden_size=hidden_size)

        self.correlation_mat = nn.Parameter(torch.FloatTensor(hidden_size,hidden_size),requires_grad=True)
        nn.init.orthogonal_(self.correlation_mat.data, gain=1)
        self.fc = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        self.critic_linear = nn.Sequential(
                init_(nn.Linear(hidden_size * 2, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, 1)))
        self.train()

    def forward(self, share_inputs, inputs, agent_i, rnn_hxs, masks, agent_num=2, adv_num=2, good_num = 2, box_num = 2, ramp_num = 1):
        """
        share_inputs: [batch_size, obs_dim*agent_num]
        inputs: [batch_size, obs_dim]
        """
        batch_size = inputs.shape[0]
        obs_dim = inputs.shape[-1]
        hidden_actor = self.actor(inputs, adv_num, good_num, box_num, ramp_num)
        f_ii = self.encoder(inputs, adv_num, good_num, box_num, ramp_num)
        obs_beta_ij = torch.matmul(f_ii.view(batch_size,1,-1), self.correlation_mat)
        obs_encoder = []
        beta = []
        for i in range(good_num + adv_num):
            if i != agent_i:
                f_ij = self.encoder(share_inputs[:, i*obs_dim:(i+1)*obs_dim], adv_num, good_num, box_num, ramp_num)     #[batch_size, hidden_size]
                obs_encoder.append(f_ij)
        obs_encoder = torch.stack(obs_encoder,dim = 1)    #(batch_size,n_agents-1,eb_dim)
        beta = torch.matmul(obs_beta_ij, obs_encoder.permute(0,2,1)).squeeze(1)
        alpha = F.softmax(beta,dim = 1).unsqueeze(2)
        vi = torch.mul(alpha,obs_encoder)
        vi = torch.sum(vi,dim = 1)
        gi = self.fc(f_ii)
        value = self.critic_linear(torch.cat([gi, vi], dim=1))

        return value, hidden_actor, rnn_hxs
