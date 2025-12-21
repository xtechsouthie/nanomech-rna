import yaml
import torch
import numpy as np
from pathlib import Path
import argparse

from imitation.data_GIN import create_dataloaders
from RNA_RL.network_classes import create_network
from imitation.train_gin import BCTrainer

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(config_path: str):

    config = load_config(config_path)
    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_folder=config['data']['folder'],
        batch_size=config['training']['batch_size'],
        test_size=config['data']['test_size'],
        val_size=config['data']['val_size'],
        num_workers=config['data']['num_workers']
    )

    print(f"length of data: train: {len(train_loader.dataset)},\
           val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")
    
    print("\ncreating network...")
    network = create_network(config=config['network'])

    train_cfg = config['training']
    trainer = BCTrainer(
        network=network,
        device=device,
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        location_weight=train_cfg['location_weight'],
        mutation_weight=train_cfg['mutation_weight'],
        grad_clip=train_cfg['grad_clip']
    )

    print(f"\ntraining for {train_cfg['num_epochs']} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=train_cfg['num_epochs'],
        save_dir=config['checkpoint']['save_dir'],
        patience=train_cfg['patience']
    )

    print('\nEval on the test set...')
    trainer.load_checkpoint(f"{config['checkpoint']['save_dir']}/best_model.pt")
    test_metrics = trainer.validate(test_loader)

    print(f"\nTest results:")
    print(f"    loss: {test_metrics['total_loss']:.4f}")
    print(f"    location acc: {test_metrics['location_acc']:.3f} ({test_metrics['location_acc']*100:.1f}%)")
    print(f"    mutation acc: {test_metrics['mutation_acc']:.3f} ({test_metrics['mutation_acc']*100:.1f}%)")

    return history, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_bc.yaml')
    args = parser.parse_args()

    main(args.config)
    