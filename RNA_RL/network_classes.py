import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from typing import Tuple, Optional, Dict
import logging 

logger = logging.getLogger(__name__)

# class MLP(nn.Module):

#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Linear(in_channels, out_channels),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_channels, out_channels)
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         return self.layers(x)
    
class GINLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, residual: bool = True):
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )

        self.gin = GINEConv(mlp, edge_dim=edge_dim)
        self.residual = residual and (in_channels == out_channels)

    def forward(self, x:torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.gin(x, edge_index, edge_attr)

        if self.residual:
            out = out + x
        
        return out
    
class FeatureExtractorGNN(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, edge_dim: int = 4,
                  num_layers: int = 2, dropout: int = 0.3, residual: bool = True):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers == 1:
            dims = [in_dim, out_dim]
        else: 
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_residual = residual and (dims[i] == dims[i+1])
            self.layers.append(
                GINLayer(dims[i], dims[i + 1], edge_dim, residual=use_residual)
            )

        logger.info(f"The gnn is implemented with num_layers: {num_layers}, and dims: {dims}, residual: {residual}")

    def forward(self, x:torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            if self.dropout > 0 and self.training:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
    

class LocationActor(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace =True),
            nn.Linear(hidden_dim, 1)  
        )


    def forward(self, x: torch.Tensor)-> torch.Tensor:

        return self.net(x).squeeze(-1)
    
    
class MutationActor(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 16, num_bases: int = 4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace =True),
            nn.Linear(hidden_dim, num_bases)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)
    

class LocationCritic(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace =True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x:torch.Tensor, batch: torch.Tensor) -> torch.Tensor:

        x_pooled = global_add_pool(x, batch) #from node level to graph level -> output = [B, in_dim] 

        return self.net(x_pooled).squeeze(-1) # [B]
    

class MutationCritic(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 16):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace =True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x).squeeze(-1) #value : [B]
    

class RNADesignNetwork(nn.Module):

    def __init__(self, in_dim: int = 9, hidden_dim: int = 32, embed_dim: int = 16, 
                 edge_dim: int = 4, num_gnn_layers: int = 2, actor_hidden: int = 16,
                 critic_hidden: int = 16, num_bases:int =4, dropout: float = 0.3, residual: bool = False):
        
        super().__init__()

        self.feature_extractor = FeatureExtractorGNN(
            in_dim=in_dim, 
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            edge_dim=edge_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            residual=residual
        )

        self.location_actor = LocationActor(embed_dim, actor_hidden)
        self.location_critic = LocationCritic(embed_dim, critic_hidden)

        self.mutation_actor = MutationActor(embed_dim, actor_hidden, num_bases)
        self.mutation_critic = MutationCritic(embed_dim, critic_hidden)

        self.embed_dim = embed_dim
        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"RNADesignNetwork initialized:  in={in_dim}, hidden={hidden_dim},\n embed={embed_dim}, layers={num_gnn_layers}")
        logger.info(f"total_params: {total_params:,}, trainable params: {trainable_params:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward_bc(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                 edge_attr: torch.Tensor, batch: torch.Tensor, ptr: torch.Tensor,
                 target_locations: torch.Tensor) -> Dict[str, torch.Tensor]:

        embeddings = self.feature_extractor(node_features, edge_index, edge_attr) #[N, embed_dim]

        location_logits = self.location_actor(embeddings) #[N]

        selected_embeddings = embeddings[target_locations] # [B]

        mutation_logits = self.mutation_actor(selected_embeddings)

        return {
            'location_logits':  location_logits, # [N]
            'mutation_logits': mutation_logits, # [B, 4]
            'embeddings': embeddings #[N, embed_dim]
        }

    def forward_inference(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                          batch: torch.Tensor, ptr: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        embeddings = self.feature_extractor(node_features, edge_index, edge_attr)

        location_logits = self.location_actor(embeddings)
        batch_size = ptr.size(0) - 1
        sampled_locations = []
        location_log_probs = torch.zeros_like(location_logits)

        for i in range(batch_size):
            start, end = ptr[i].item(), ptr[i + 1].item()
            graph_logits = location_logits[start: end]
            graph_probs = F.softmax(graph_logits, dim = 0)
            location_log_probs[start: end] = F.log_softmax(graph_logits, dim=0)
            
            # sample location
            local_idx = torch.multinomial(graph_probs, 1).item()
            global_idx = start + local_idx
            sampled_locations.append(global_idx)
        
        sampled_locations = torch.tensor(sampled_locations, device=embeddings.device, dtype=torch.long)

        selected_embeddings = embeddings[sampled_locations]
        mutation_logits = self.mutation_actor(selected_embeddings)
        mutation_probs = F.softmax(mutation_logits, dim=-1)

        location_value = self.location_critic(embeddings, batch)
        mutation_value = self.mutation_critic(selected_embeddings)


        return {
            'sampled_locations': sampled_locations,  # [B]
            'location_log_probs': location_log_probs,  # [N]
            'mutation_probs': mutation_probs,  # [B, 4]
            'mutation_logits': mutation_logits, # [B, 4]
            'location_value':  location_value, # [B]
            'mutation_value':  mutation_value,  # [B]
            'embeddings': embeddings #[ N, embed_dims]
        }
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, 
                batch: torch.Tensor, ptr: torch.Tensor,
                 target_locations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        if target_locations is not None:
            return self.forward_bc(
                node_features, edge_index, edge_attr, 
                batch, ptr, target_locations
            )
        else:
            return self.forward_inference(
                node_features, edge_index, edge_attr,
                batch, ptr
            )

def get_default_config() -> Dict:
    return {
        'in_dim': 8,
        'hidden_dim': 32,
        'embed_dim': 16,
        'edge_dim': 4,
        'num_gnn_layers':  2,
        'actor_hidden': 16,
        'critic_hidden': 16,
        'num_bases': 4,
        'dropout': 0.3
    }


def create_network(config: Optional[Dict] = None) -> RNADesignNetwork:
    if config is None:
        config = get_default_config()
    return RNADesignNetwork(**config) 

    