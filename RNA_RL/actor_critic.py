import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import global_add_pool
from typing import NamedTuple
import logging 


from RNA_RL.network_classes import (
    FeatureExtractorGNN,
    LocationActor,
    MutationActor,
    LocationCritic,
    MutationCritic
)

logger = logging.getLogger(__name__)

class ActionOutput(NamedTuple):
    locations: torch.Tensor
    mutations: torch.Tensor
    location_log_probs: torch.Tensor
    mutation_log_probs: torch.Tensor
    location_entropy: torch.Tensor
    mutation_entropy: torch.Tensor

class ValueOutput(NamedTuple):
    location_values: torch.Tensor
    mutation_values: torch.Tensor

class ActorCritic(nn.Module):

    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, embed_dim: int = 32,
                 edge_dim: int = 4, num_gnn_layers: int = 3, actor_hidden: int = 32, 
                 critic_hidden: int = 32, num_bases: int = 4, dropout: float = 0.25):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_bases = num_bases

        self.feature_extractor = FeatureExtractorGNN(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=embed_dim, edge_dim=edge_dim,
            num_layers=num_gnn_layers, dropout=dropout
        )

        self.location_actor = LocationActor(embed_dim, actor_hidden)
        self.mutation_actor = MutationActor(embed_dim, actor_hidden, num_bases)

        self.location_critic = LocationCritic(embed_dim, critic_hidden)
        self.mutation_critic = MutationCritic(embed_dim, critic_hidden)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        actor_params = sum(p.numel() for p in self.actor_parameters())
        critic_params = sum(p.numel() for p in self.critic_parameters())

        logger.info(f"Actor Critic initialized: ")
        logger.info(f"  input_dims: {in_dim}, hidden: {hidden_dim}, embed: {embed_dim}")
        logger.info(f"  GNN layers: {num_gnn_layers}, dropout: {dropout}")
        logger.info(f"  total params: {total_params}")
        logger.info(f"  actor params: {actor_params}, critic params: {critic_params}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn. init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity ='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)

    def actor_parameters(self):
        params = list(self.feature_extractor.parameters()) #i added the gnn in here too
        params += list(self.location_critic.parameters())
        params += list(self.mutation_actor.parameters())
        return params 
    
    def critic_parameters(self):
        params = list(self.location_critic.parameters())
        params += list(self.mutation_critic.parameters())
        return params
    
    def backbone_parameters(self):
        return list(self.feature_extractor.parameters())
        

    def forward(self, x, edge_index, edge_attr, batch, ptr, deterministic: bool =False):

        embeddings = self.feature_extractor(x, edge_index, edge_attr)

        batch_size = ptr.size(0) - 1
        device = x.device

        location_logits = self.location_actor(embeddings)

        locations = torch.zeros(batch_size, dtype=torch.long, device=device)

        location_log_probs = torch.zeros(batch_size, device=device)
        location_entropy = torch.zeros(batch_size, device=device)

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            graph_logits = location_logits[start:end]

            dist = Categorical(logits=graph_logits)

            if deterministic:
                local_idx = graph_logits.argmax()
            else:
                local_idx = dist.sample()

            global_idx = start + local_idx.item()
            locations[i] = global_idx #idhar yaad rakhiyo global kardiya hai local nai, badme local mat lena
            location_log_probs[i] = dist.log_prob(local_idx)
            location_entropy[i] = dist.entropy()

        selected_embeddings = embeddings[locations]

        mutation_logits = self.mutation_actor(selected_embeddings)
        mutation_dist = Categorical(mutation_logits)

        if deterministic:
            mutations = mutation_dist.argmax(dim=-1)
        else:
            mutations = mutation_dist.sample()

        mutation_log_probs = mutation_dist.log_prob(mutations)
        mutation_entropy = mutation_dist.entropy()

        location_values = self.location_critic(embeddings, batch)
        mutation_values = self.mutation_critic(selected_embeddings)

        actions = ActionOutput(
            locations = locations,
            mutations = mutations,
            location_log_probs=location_log_probs,
            mutation_log_probs=mutation_log_probs,
            location_entropy=location_entropy,
            mutation_entropy=mutation_entropy
        )

        values = ValueOutput(
            location_values=location_values,
            mutation_values=mutation_values
        )

        return actions, values, embeddings
    
    def evaluate_actions(self, x, edge_index, edge_attr, batch, ptr, locations, mutations):

        embeddings = self.feature_extractor(x, edge_index, edge_attr)

        batch_size = ptr.size(0) - 1
        device = x.device

        location_logits = self.location_actor(embeddings)

        location_log_probs = torch.zeros(batch_size, device =device)
        location_entropy = torch.zeros(batch_size, device = device)

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            graph_logits = location_logits[start:end]

            dist = Categorical(logits = graph_logits)
            local_idx = locations[i] - start

            location_log_probs[i] = dist.log_prob(local_idx)
            location_entropy[i] = dist.entropy()

        selected_embeddings = embeddings[locations]

        mutation_logits = self.mutation_actor(selected_embeddings)
        mutation_dist = Categorical(logits = mutation_logits)

        mutation_log_probs = mutation_dist.log_prob(mutations)
        mutation_entropy = mutation_dist.entropy()

        location_values = self.location_critic(embeddings, batch)
        mutation_values = self.mutation_critic(selected_embeddings)

        return (
            location_log_probs, mutation_log_probs,
            location_entropy, mutation_entropy,
            location_values, mutation_values
        )
    

    def get_values(self, x, edge_index, edge_attr, batch, ptr):

        embeddings = self.feature_extractor(x, edge_index, edge_attr)

        location_values = self.location_critic(embeddings, batch)

        #isme we dont have the specific locations, so we use global_add_pool, ek graph level embedding mil jaegi
        graph_embeddings = global_add_pool(embeddings, batch)
        mutation_values = self.mutation_critic(graph_embeddings)

        return location_values, mutation_values
    
