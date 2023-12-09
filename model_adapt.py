
import sys
sys.path.insert(0, "composite")
import copy

import torch
import numpy as np
from typing import Optional

from models import Discriminator, RunningMeanStd, DiagonalPopArt
from models import ACModel as ACModelBase


class ACModel(torch.nn.Module):

    def __init__(self, state_dim: int, act_dim: int, goal_dim: int=0, value_dim: int=1, 
        normalize_value: bool=True,
        init_mu:Optional[torch.Tensor or float]=None, init_sigma:Optional[torch.Tensor or float]=None,
        meta_goal_dim: int=0
    ):
        super().__init__()
        self.state_dim = state_dim
        
        self.goal_dim_actor = goal_dim
        self.goal_dim_critic = goal_dim + meta_goal_dim

        self.actor = ACModelBase.Actor(state_dim, act_dim, self.goal_dim_actor, init_mu=init_mu, init_sigma=init_sigma)
        self.critic = ACModelBase.Critic(state_dim, self.goal_dim_critic, value_dim)
        self.actor_ob_normalizer = RunningMeanStd(state_dim, clamp=5.0)
        self.critic_ob_normalizer = self.actor_ob_normalizer
        self.ob_normalizer = [self.actor_ob_normalizer]
        if isinstance(self.critic_ob_normalizer, torch.nn.ModuleList):
            self.ob_normalizer.extend(self.critic_ob_normalizer)
        if normalize_value:
            self.value_normalizer = DiagonalPopArt(value_dim, 
                self.critic.mlp[-1].weight, self.critic.mlp[-1].bias)
        else:
            self.value_normalizer = None
    
    def observe(self, obs, norm=True):
        if self.goal_dim_critic > 0:
            s = obs[:, :-self.goal_dim_critic]
            g = obs[:, -self.goal_dim_critic:]
        else:
            s = obs
            g = None
        s = s.view(*s.shape[:-1], -1, self.state_dim)
        return [normalizer(s) for normalizer in self.ob_normalizer] if norm else s, g

    def eval_(self, s, seq_end_frame, g, unnorm):
        v = self.critic(s[-1], seq_end_frame, g)
        if unnorm and self.value_normalizer is not None:
            v = self.value_normalizer(v, unnorm=True)
        return v

    def act(self, obs, seq_end_frame, stochastic=None, unnorm=False):
        if stochastic is None:
            stochastic = self.training
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, None if g is None else g[:, :self.goal_dim_actor])
        if stochastic:
            a = pi.sample()
            lp = pi.log_prob(a)
            # if g is not None:
            #     g = g[...,:self.goal_dim_critic]
            return a, self.eval_(s, seq_end_frame, g, unnorm), lp
        else:
            return pi.mean


    def evaluate(self, obs, seq_end_frame, unnorm=False):
        # no meta_goal passed with obs
        # self.goal_dim, self.goal_dim_critic = self.goal_dim_critic, self.goal_dim
        s, g = self.observe(obs)
        # self.goal_dim, self.goal_dim_critic = self.goal_dim_critic, self.goal_dim
        # if g is not None:
            # g = g[...,:self.goal_dim_critic]
        return self.eval_(s, seq_end_frame, g, unnorm)
    
    def forward(self, obs, seq_end_frame, unnorm=False):
        s, g = self.observe(obs)
        pi = self.actor(s, seq_end_frame, None if g is None else g[:, :self.goal_dim_actor])
        return pi, self.eval_(s, seq_end_frame, g, unnorm)


class MapCNN(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, 5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Conv2d(128, 256, 3),
            torch.nn.Flatten()
        )
    
    def forward(self, m):
        return self.cnn(m)


class AdaptNet(torch.nn.Module):
    def __init__(self, meta_model, g_dim=0):
        super().__init__()
        actor_ob_normalizer = copy.deepcopy(meta_model.actor_ob_normalizer)
        for normalizer in meta_model.ob_normalizer:
            normalizer.reset_counter()
        meta_model.ob_normalizer.insert(0, actor_ob_normalizer)

        meta_policy = meta_model.actor
        for n, p in meta_policy.named_parameters():
            p.requires_grad = False

        input_size = meta_policy.rnn.input_size
        hidden_size = meta_policy.rnn.hidden_size
        num_layers = meta_policy.rnn.num_layers
        batch_first = meta_policy.rnn.batch_first

        self.meta_policy = meta_policy
        self.rnn = meta_policy.rnn.__class__(input_size, hidden_size, num_layers, batch_first=batch_first)
        
        self.embed = torch.nn.Linear(meta_policy.mlp[0].in_features+g_dim, meta_policy.mlp[0].in_features)
        
        ia_layer = lambda in_dim, out_dim: torch.nn.Linear(in_dim, out_dim)
        self.ia = torch.nn.ModuleList([ia_layer(op.in_features, op.out_features) 
            if isinstance(op, torch.nn.Linear) else torch.nn.Identity() for op in meta_policy.mlp])
        for e in self.ia:
            if isinstance(e, torch.nn.Identity): continue
            if isinstance(e, torch.nn.Sequential): e = e[-1]
            for p in e.parameters():
                torch.nn.init.zeros_(p)
    
        for p, p_ in zip(self.rnn.parameters(), meta_policy.rnn.parameters()):
            p.data.copy_(p_.data)
        for p in self.embed.parameters():
            torch.nn.init.zeros_(p)
        
        self.g = None
    
    def forward(self, s, seq_end_frame, g=None):
        s, s_ = s[0], s[1]
        
        n_inst = s.size(0)
        if n_inst > self.meta_policy.all_inst.size(0):
            self.meta_policy.all_inst = torch.arange(n_inst, 
                dtype=seq_end_frame.dtype, device=seq_end_frame.device)
        ind = (self.meta_policy.all_inst[:n_inst], torch.clip(seq_end_frame, max=s.size(1)-1))
        s_, _ = self.rnn(s_)
        s_ = s_[ind]
        s, _ = self.meta_policy.rnn(s)
        s = s[ind]

        if g is not None:
            s = torch.cat((s, g), -1)
            if self.g is None:
                s_ = torch.cat((s_, g), -1)
            else:
                s_ = torch.cat((s_, g, self.g), -1)
        elif self.g is not None:
            s_ = torch.cat((s_, self.g), -1)

        if isinstance(self.embed, torch.nn.ModuleList):
            s_ = [embed(s_) for embed in self.embed]
        else:
            s_ = self.embed(s_)
        s = s + s_
        for j, op in enumerate(self.meta_policy.mlp):
            embed = self.ia[j]
            if isinstance(embed, torch.nn.Identity):
                s = op(s)
            else:
                s = op(s) + embed(s)
        mu = self.meta_policy.mu(s)
        sigma = torch.exp(self.meta_policy.log_sigma(s)) + 1e-8
        return torch.distributions.Normal(mu, sigma)
