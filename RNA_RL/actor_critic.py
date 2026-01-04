import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from typing import NamedTuple, Optional
import logging 
import copy

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
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity ='relu')
                if m.bias is not None: 
                    nn.init.constant_(m.bias, 0)

    def actor_parameters(self):
        params = list(self.feature_extractor.parameters()) #i added the gnn in here too
        params += list(self.location_actor.parameters())
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
        global_locations = torch.zeros(batch_size, dtype=torch.long, device=device)

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


            locations[i] = local_idx
            global_locations[i] = start + local_idx.item()
            location_log_probs[i] = dist.log_prob(local_idx)
            location_entropy[i] = dist.entropy()

            # if i == 0:
            #     print(f"graph logits:   {graph_logits}")
            #     print(f"selected local idx from the logits: {local_idx.item()}")
            #     print(f"selected global idx from the logits: {start + local_idx.item()}")
            #     print(f"embeddings shape: {embeddings.shape}")
            #     print(f"ptr shape: {ptr.shape}")
            #     print(f"ptr: {ptr}")


        selected_embeddings = embeddings[global_locations]

        mutation_logits = self.mutation_actor(selected_embeddings)
        mutation_dist = Categorical(logits=mutation_logits)

        if deterministic:
            mutations = mutation_logits.argmax(dim=-1)
        else:
            mutations = mutation_dist.sample()

        mutation_log_probs = mutation_dist.log_prob(mutations)
        mutation_entropy = mutation_dist.entropy()


        location_values = self.location_critic(embeddings, batch)
        mutation_values = self.mutation_critic(selected_embeddings)

        #print(f"locations: {locations}, mutations: {mutations}")

        actions = ActionOutput(
            locations = locations,
            mutations = mutations,
            location_log_probs = location_log_probs,
            mutation_log_probs = mutation_log_probs,
            location_entropy = location_entropy,
            mutation_entropy =mutation_entropy
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
        global_locations = torch.zeros(batch_size, dtype=torch.long, device=device)


        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i+1].item()
            graph_logits = location_logits[start:end]

            dist = Categorical(logits = graph_logits)
            local_idx = locations[i]

            num_nodes = end - start

            if local_idx < 0 or local_idx >= num_nodes:
                logger.error(f"invalid local idx: {local_idx}, num_nodes: {num_nodes}")
                local_idx = torch.clamp(local_idx, 0, num_nodes - 1)

            computed_log_prob = dist.log_prob(local_idx)
            location_log_probs[i] = computed_log_prob
            location_entropy[i] = dist.entropy()
            global_locations[i] = start + local_idx.item()

        selected_embeddings = embeddings[global_locations]


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
    
    def load_pretrained_bc(self, checkpoint_path, load_backbone, load_actors, freeze_backbone, 
                           freeze_actors, strict=False):
        
        logger.info(f"Loading pretrained behaviour cloning weights from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        bc_state_dict = checkpoint['model_state_dict']

        current_state = self.state_dict()

        loaded_back_keys = []
        skipped_back_keys = []

        if load_backbone:
            backbone_keys = [k for k in bc_state_dict.keys() if k.startswith('feature_extractor')]
            for key in backbone_keys:
                if key in current_state and bc_state_dict[key].shape == current_state[key].shape:
                    current_state[key].copy_(bc_state_dict[key])
                    loaded_back_keys.append(key)
                else:
                    skipped_back_keys.append(key)
            logger.info(f"loaded {len(loaded_back_keys)} GNN keys and skipped {len(skipped_back_keys)} GNN keys")

        loaded_act_keys = []
        skipped_act_keys = []

        if load_actors:
            actor_keys = [k for k in bc_state_dict.keys() if k.startswith('location_actor') or k.startswith('mutation_actor')]
            for key in actor_keys:
                if key in current_state and bc_state_dict[key].shape == current_state[key].shape:
                    current_state[key].copy_(bc_state_dict[key])
                    loaded_act_keys.append(key)
                else:
                    skipped_act_keys.append(key)
            logger.info(f"loaded {len(loaded_act_keys)} actor keys and skipped {len(skipped_act_keys)} actor keys")

        self.load_state_dict(current_state)

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("feature extractor parameters frozen")

        if freeze_actors:
            for param in self.location_actor.parameters():
                param.requires_grad = False
            for param in self.mutation_actor.parameters():
                param.requires_grad = False
            logger.info("actor parameters are frozen")


        loaded_back_keys.extend(loaded_act_keys)
        skipped_back_keys.extend(skipped_act_keys)

        info = {
            'loaded_keys': loaded_back_keys,
            'skipped_keys': skipped_back_keys,
            'bc_epoch': checkpoint.get('epoch', -1),
            'bc_metrics': checkpoint.get('metrics', {}),
            'backbone_frozen': freeze_backbone,
            'actors_frozen': freeze_actors
        }

        logger.info(f"loaded {len(loaded_back_keys)} parameters from BC checkpoint (epoch {info['bc_epoch']})")
        if skipped_back_keys:
            logger.warning(f"skipped {len(skipped_back_keys)} keys (shape mismatch or not found)")
        
        return info
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    def freeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        logger.info(f"backbone parameters frozen")

    def unfreeze_backbone(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        logger.info("backbone parameters unfrozen")

    def get_trainable_params_info(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        total = trainable + frozen

        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_pct': (100 * trainable / total) if total > 0 else 0
        }
    

def create_actor_critic(in_dim: int = 8, hidden_dim: int = 64,
                        embed_dim: int = 32, edge_dim: int = 4, num_gnn_layers: int = 3,
                        dropout: float = 0.2, pretrained_path: Optional[str] = None, 
                        freeze_pretrained: bool = False, device: torch.device = torch.device('cpu')):
    
    model = ActorCritic(in_dim = in_dim, hidden_dim= hidden_dim,
                        embed_dim= embed_dim, edge_dim= edge_dim,
                        num_gnn_layers= num_gnn_layers, dropout= dropout).to(device)
    
    if pretrained_path is not None:
        model.load_pretrained_bc(
            pretrained_path,
            load_backbone= True,
            load_actors= True,
            freeze_backbone= freeze_pretrained,
            freeze_actors= False
        )

    return model

