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
        
        if self.ptr >= self.size:
            return False

        self.state.append(state.clone())
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
        return True

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
        self.mutation_adv.extend(
            discount_cumsum(mut_deltas, self.gamma * self.lam).tolist()
        )
        
        self.mutation_returns.extend(
            discount_cumsum(mut_rews, self.gamma)[:-1].tolist()
        )

        self.path_start_idx = self.ptr

    def get(self, device):

        actual_size = len(self.location_adv)

        if actual_size == 0:
            raise ValueError("Buffer is empty, cannot get data...")

        loc_adv = np.array(self.location_adv)
        mut_adv = np.array(self.mutation_adv)

        loc_adv = (loc_adv - loc_adv.mean()) / (loc_adv.std() + 1e-7)
        mut_adv = (mut_adv - mut_adv.mean()) / (mut_adv.std() + 1e-7)

        data = dict(
            states=self.state.copy(),
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
        self.state.clear()
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
    
class PPO:

    def __init__(self, actor_critic, lr_actor = 1e-4, lr_critic = 1e-3, lr_backbone = 1e-4,
                 gamma = 0.99, lam = 0.95, clip_ratio = 0.2, target_kl = 0.01, entropy_coef = 0.01,
                 value_coef = 0.5, max_grad_norm = 0.5, train_actor_iters = 10, train_critic_iters = 10,
                 seprate_backbone_lr = True, device = torch.device('cpu')):
        
        self.ac = actor_critic
        self.device = device

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.train_actor_iters = train_actor_iters
        self.train_critic_iters = train_critic_iters

        if seprate_backbone_lr:
            self.actor_optimizer = Adam([
                {'params': self.ac.feature_extractor.parameters(), 'lr': lr_backbone},
                {'params': self.ac.location_actor.parameters(), 'lr': lr_actor},
                {'params': self.ac.mutation_actor.parameters(), 'lr': lr_actor},
            ])
        else:
            self.actor_optimizer = Adam(self.ac.actor_parameters(), lr=lr_actor)

        self.critic_optimizer = Adam(self.ac.critic_parameters(), lr= lr_critic)

        logger.info(f"PPO inititalized on {device}")
        logger.info(f"  actor lr: {lr_actor}, critic lr: {lr_critic}, backbone_lr: {lr_backbone}")
        logger.info(f"  seprate backbone lr: {seprate_backbone_lr}, clip ratio: {clip_ratio}, target kl: {target_kl}")

    def select_action(self, state, deterministic = False):

        self.ac.eval()

        with torch.no_grad():

            batch = Batch.from_data_list([state])

            actions, values, _ = self.ac(
                x = batch.x.float().to(self.device),
                edge_index = batch.edge_index.to(self.device),
                edge_attr = batch.edge_attr.float().to(self.device),
                batch = batch.batch.to(self.device),
                ptr = batch.ptr.to(self.device),
                deterministic = deterministic,
            )

            return (
                actions.locations[0].item(),
                actions.mutations[0].item(),
                actions.location_log_probs[0].item(),
                actions.mutation_log_probs[0].item(),
                values.location_values[0].item(),
                values.mutation_values[0].item()
            )
        
    def get_value(self, state):

        self.ac.eval()

        with torch.no_grad():
            batch = Batch.from_data_list([state])

            loc_val, mut_val = self.ac.get_values(
                x = batch.x.float().to(self.device),
                edge_index = batch.edge_index.to(self.device),
                edge_attr = batch.edge_attr.float().to(self.device),
                batch = batch.batch.to(self.device),
                ptr = batch.ptr.to(self.device),
            )

            return loc_val[0].item(), mut_val[0].item()
        
    def compute_loss_pi(self, batch, data):

        loc_log_probs, mut_log_probs, loc_entropy, mut_entropy, _, _ = self.ac.evaluate_actions(
            x = batch.x.float().to(self.device),
            edge_index = batch.edge_index.to(self.device),
            edge_attr = batch.edge_attr.float().to(self.device),
            batch = batch.batch.to(self.device),
            ptr = batch.ptr.to(self.device),
            locations = data['locations'],
            mutations = data['mutations']
        )

        loc_ratio = torch.exp(loc_log_probs - data['loc_log_probs'])
        loc_adv = data['loc_advantages']
        loc_clip = torch.clamp(loc_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_loc = -(torch.min(loc_ratio * loc_adv, loc_clip * loc_adv)).mean()

        mut_ratio = torch.exp(mut_log_probs - data['mut_log_probs'])
        mut_adv = data['mut_advantages']
        mut_clip = torch.clamp(mut_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        loss_mut = -(torch.min(mut_ratio * mut_adv, mut_clip * mut_adv)).mean()

        entropy = (loc_entropy.mean() + mut_entropy.mean()) / 2

        loss_pi = loss_loc + loss_mut - self.entropy_coef * entropy #isme entropy dalke and nikalke dono test karle better kya aata hai

        with torch.no_grad():
            loc_log_ratio = loc_log_probs - data['loc_log_probs']
            mut_log_ratio = mut_log_probs - data['mut_log_probs']
            
            approx_kl = 0.5 * (loc_log_ratio.pow(2).mean() + mut_log_ratio.pow(2).mean()).item()
        
            loc_clipfrac = ((loc_ratio - 1).abs() > self.clip_ratio).float().mean().item()
            mut_clipfrac = ((mut_ratio - 1).abs() > self.clip_ratio).float().mean().item()
        
        pi_info = dict(
            kl = approx_kl,
            entropy = entropy.item(),
            loc_clip_frac = loc_clipfrac,
            mut_clip_frac = mut_clipfrac,
            loc_loss = loss_loc.item(),
            mut_loss = loss_mut.item()
        )
        
        return loss_pi, pi_info
    
    def compute_loss_v(self, batch, data):

        _, _, _, _, loc_values, mut_values = self.ac.evaluate_actions(
            x = batch.x.float().to(self.device),
            edge_index = batch.edge_index.to(self.device),
            edge_attr = batch.edge_attr.float().to(self.device),
            batch = batch.batch.to(self.device),
            ptr = batch.ptr.to(self.device),
            locations = data['locations'],
            mutations = data['mutations']
        )

        loc_loss = F.mse_loss(loc_values, data['loc_returns'])
        mut_loss = F.mse_loss(mut_values, data['mut_returns'])

        return (loc_loss + mut_loss)       


    def update(self, buf):

        data = buf.get(self.device)

        batch = Batch.from_data_list(data['states'])

        # self.ac.train()

        pi_info = {}
        for i in range(self.train_actor_iters):

            self.actor_optimizer.zero_grad()

            self.ac.eval()
            loss_pi, pi_info = self.compute_loss_pi(batch, data)

            if pi_info['kl'] > 1.5 * self.target_kl:
                logger.info(f"early stopping at step {i} due to reaching max kl")
                break


            self.ac.train()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        pi_info['stop_iter'] = i

        for i in range(self.train_critic_iters):
            self.critic_optimizer.zero_grad()
            self.ac.eval()
            loss_v = self.compute_loss_v(batch, data)
            self.ac.train()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        pi_info['critic_loss'] = loss_v.item()

        return pi_info
    
    def save(self, path):

        torch.save({
            'actor_critic': self.ac.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
        logger.info(f"model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(checkpoint['actor_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        logger.info(f"model loaded from path: {path}")

        



        

        



