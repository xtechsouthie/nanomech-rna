import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import logging
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import numpy as np

from imitation.data_GIN import create_dataloaders, add_global_locations
from RNA_RL.network_classes import RNADesignNetwork, create_network, get_default_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BCTrainer:

    def __init__(self, network: RNADesignNetwork, device: torch.device,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5,
                location_weight: float = 1.0, mutation_weight: float = 1.0,
                grad_clip: float = 1.0):
        
        self.network = network.to(device)
        self.device = device
        self.location_weight = location_weight
        self.mutation_weight = mutation_weight
        self.grad_clip = grad_clip

        self.optimizer = optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )

        # self.location_loss_fn = nn.CrossEntropyLoss()
        # self.mutation_loss_fn = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_loc_acc': [], 'val_loc_acc': [],
            'train_mut_acc': [], 'val_mut_acc': [], 'lr': []
        }

        logger.info(f"BCTrainer initialized on {device}")

    def _compute_loss_and_metrics(self, outputs: Dict[str, torch.Tensor], 
                                    batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        location_logits = outputs['location_logits']
        mutation_logits = outputs['mutation_logits']

        target_locations = batch.y_location_local
        target_mutations = batch.y_mutation.squeeze(-1)
        ptr = batch.ptr

        batch_size = ptr.size(0) - 1

        location_loss = 0.0
        location_correct = 0

        for i in range(batch_size):

            start, end = ptr[i].item(), ptr[i+1].item()
            graph_logits = location_logits[start:end]
            graph_size = end - start

            target_local = target_locations[i].item()

            if target_local >= graph_size:
                logger.info(f"error: target location - {target_local} > graph size - {graph_size}")
                target_local = min(max(target_local, 0), graph_size - 1)

            loss_i = F.cross_entropy(
                graph_logits.unsqueeze(0),
                torch.tensor([target_local], device=self.device)
            )

            location_loss += loss_i

            pred_local = graph_logits.argmax().item()
            if pred_local == target_local:
                location_correct += 1

        location_loss = location_loss / batch_size
        location_acc = location_correct / batch_size

        mutation_loss = F.cross_entropy(mutation_logits, target_mutations)
        pred_mutations = mutation_logits.argmax(dim = 1)
        mutation_acc = (pred_mutations == target_mutations).float().mean().item()

        total_loss = ((self.location_weight * location_loss) + (self.mutation_weight * mutation_loss))

        metrics = {
            'total_loss': total_loss.item(),
            'location_loss': location_loss.item(),
            'mutation_loss': mutation_loss.item(),
            'location_acc': location_acc,
            'mutation_acc': mutation_acc,
        }

        return total_loss, metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:

        self.network.train()

        total_metrics = {
            'total_loss': 0, 'location_loss': 0, 'mutation_loss': 0,
            'location_acc': 0, 'mutation_acc': 0
        }
        num_batches = 0

        for batch in dataloader:

            batch = add_global_locations(batch)
            batch = batch.to(self.device)

            outputs = self.network(
                node_features=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                ptr=batch.ptr,
                target_locations=batch.y_location_global
            )

            loss, metrics = self._compute_loss_and_metrics(outputs, batch)

            self.optimizer.zero_grad()

            loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            
            self.optimizer.step()

            for k in total_metrics:
                total_metrics[k] += metrics[k]

            num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:

        self.network.eval()

        total_metrics = {
            'total_loss': 0, 'location_loss': 0, 'mutation_loss': 0,
            'location_acc': 0, 'mutation_acc': 0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:

                batch = add_global_locations(batch)
                batch = batch.to(self.device)

                outputs = self.network(
                    node_features=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    ptr=batch.ptr,
                    target_locations=batch.y_location_global
                )

                _, metrics = self._compute_loss_and_metrics(outputs, batch)

                for k in total_metrics:
                    total_metrics[k] += metrics[k]

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}

    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100,
              save_dir: str = './checkpoints', patience: int = 20) -> Dict:
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0

        logging.info(f"Starting training for {num_epochs} epochs, checkpoints saved to {save_path}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['total_loss'])
            
            self.history['train_loss'].append(train_metrics['total_loss'])
            self.history['val_loss'].append(val_metrics['total_loss'])
            self.history['train_loc_acc'].append(train_metrics['location_acc'])
            self.history['val_loc_acc'].append(val_metrics['location_acc'])
            self.history['train_mut_acc']. append(train_metrics['mutation_acc'])
            self.history['val_mut_acc'].append(val_metrics['mutation_acc'])
            self.history['lr'].append(float(current_lr))
            
            epoch_time = time.time() - epoch_start
            
            logger.info(
                f"\nEPOCH --- {epoch}/{num_epochs} ({epoch_time:.1f}s)  |  "
                f"train loss: {train_metrics['total_loss']:.4f}\n"
                f"val loss: {val_metrics['total_loss']:.4f}  |  "
                f"loc acc: {val_metrics['location_acc']:.3f}\n"
                f"mut acc: {val_metrics['mutation_acc']:.3f}  |  "
                f"current learning rate: {current_lr}\n"
            )
            
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                self._save_checkpoint(save_path / 'best_model.pt', epoch, val_metrics)
                logger.info(f"---> New best model found, val loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}, cause of no new good models")
                break
            
            if epoch % 25 == 0:
                self._save_checkpoint(save_path / f'checkpoint_ep{epoch}.pt', epoch, val_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        
        self._save_checkpoint(save_path / 'final_model.pt', epoch, val_metrics)
        self._save_history(save_path / 'training_history.json')
        self._plot_curves(save_path / 'training_curves.png')
        
        return self.history
    
    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict':  self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'history': self.history
        }, path)

    def load_checkpoint(self, path: str):

        checkpoint = torch.load(path, map_location = self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def _save_history(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _plot_curves(self, path: Path):

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total loss')
        axes[0].legend()
        axes[0].grid(True, alpha = 0.3)
    
        axes[1].plot(epochs, self.history['train_loc_acc'], label='Train')
        axes[1].plot(epochs, self.history['val_loc_acc'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Location accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha = 0.3)
        
        axes[2].plot(epochs, self.history['train_mut_acc'], label='Train')
        axes[2].plot(epochs, self.history['val_mut_acc'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Mutation accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha = 0.3)
        
        plt.tight_layout()
        plt.savefig(path, dpi =150)
        plt.close()
        print(f"Saved training curves to {path}")
        logger.info(f"Saved training curves to {path}")


    @torch.no_grad()
    def test_model(self, dataloader: DataLoader) -> Dict[str, float]:

        self.network.eval()

        total_metrics = {
            'total_loss': 0, 'location_loss': 0, 'mutation_loss': 0,
            'location_acc': 0, 'mutation_acc': 0
        }

        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:

                batch = add_global_locations(batch)
                batch = batch.to(self.device)

                outputs = self.network(
                    node_features=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    ptr=batch.ptr,
                    target_locations=batch.y_location_global
                )

                _, metrics = self._compute_loss_and_metrics(outputs, batch)

                for k in total_metrics:
                    total_metrics[k] += metrics[k]

                num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}



