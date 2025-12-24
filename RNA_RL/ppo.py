import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging 

from RNA_RL.actor_critic import ActorCritic, ActionOutput, ValueOutput

logger = logging.getLogger(__name__)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class RolloutBuffer:

    def __init__(self, size, gamma=0.99, lam=0.95):

        self.size = size
        self.gamma = gamma
        self.lam = lam

        self.state = []
        self.locations = []
        self.mutations = []
        self.location_log_probs = []
        self.mutation_log_probs = []
        self.location_values = []
        self.mutation_values = []
        self.location_rewards = []
        self.mutation_rewards = []
        self.dones = []

        self.location_adv = []
        self.mutation_adv = []
        self.location_returns = []
        self.mutation_returns = []

        self.ptr, self.path_start_idx = 0, 0

    def store(self, state, location, mutation, loc_log_prob, mut_log_prob, 
              loc_value, mut_value, loc_reward, mut_reward, done):
        
        assert self.ptr < self.size

        self.state.append(state)
        self.locations.append(location)
        self.mutations.append(mutation)
        self.location_log_probs.append(loc_log_prob)
        self.mutation_log_probs.append(mut_log_prob)
        self.location_values.append(loc_value)
        self.mutation_values.append(mut_value)
        self.location_rewards.append(loc_reward)
        self.mutation_rewards.append(mut_reward)
        self.dones.append(done)
        self.ptr += 1

    def finish_path(self, last_loc_val=0, last_mut_val=0): #last loc value se bootstrap karenge if trajectory was not ended

        path_slice = slice(self.path_start_idx, self.ptr)

        loc_rews = np.append(self.location_rewards[path_slice], last_loc_val)
        loc_vals = np.append(self.location_values[path_slice], last_loc_val)

        loc_deltas = loc_rews[:-1] + self.gamma * loc_vals[1:] - loc_vals[:-1]
        self.location_adv.extend(
            discount_cumsum(loc_deltas, self.gamma * self.lam).tolist()
        )

        self.location_returns.extend(
            discount_cumsum(loc_rews, self.gamma)[:-1].tolist()
        )

        mut_rews = np.append(self.mutation_rewards[path_slice], last_mut_val)
        mut_vals = np.append(self.mutation_values[path_slice], last_mut_val)
        
        mut_deltas = mut_rews[:-1] + self.gamma * mut_vals[1:] - mut_vals[:-1]
        self.mutation_advantages.extend(
            discount_cumsum(mut_deltas, self.gamma * self.lam).tolist()
        )
        
        self.mutation_returns.extend(
            discount_cumsum(mut_rews, self.gamma)[:-1].tolist()
        )

        self.path_start_idx = self.ptr

    def get(self, device):

        assert self.ptr == self.size

        loc_adv = np.array(self.location_adv)
        mut_adv = np.array(self.mutation_adv)

        loc_adv = (loc_adv - loc_adv.mean()) / (loc_adv.std() + 1e-7)
        mut_adv = (mut_adv - mut_adv.mean()) / (mut_adv.std() + 1e-7)

        data = Dict(
            states=self.states.copy(),
            locations = torch.as_tensor(self.locations, dtype=torch.long, device=device),
            mutations = torch.as_tensor(self.mutations, dtype=torch.long, device=device),
            loc_log_probs = torch.as_tensor(self.location_log_probs, dtype=torch.float32, device=device),
            mut_log_probs = torch.as_tensor(self.mutation_log_probs, dtype=torch.float32, device=device),
            loc_advantages = torch.as_tensor(loc_adv, dtype=torch.float32, device=device),
            mut_advantages= torch.as_tensor(mut_adv, dtype=torch.float32, device=device),
            loc_returns = torch.as_tensor(self.location_returns, dtype=torch.float32, device=device),
            mut_returns = torch.as_tensor(self.mutation_returns, dtype=torch.float32, device=device),
        )

        self.clear_data()

        return data
        
    
    def clear_data(self):

        self.ptr, self.path_start_idx = 0, 0
        self.states.clear()
        self.locations.clear()
        self.mutations.clear()
        self.location_log_probs.clear()
        self.mutation_log_probs.clear()
        self.location_values.clear()
        self.mutation_values.clear()
        self.location_rewards.clear()
        self.mutation_rewards.clear()
        self.dones.clear()
        self.location_adv.clear()
        self.mutation_adv.clear()
        self.location_returns.clear()
        self.mutation_returns.clear()


    def __len__(self):
        return self.ptr
    

        

    

        

        



