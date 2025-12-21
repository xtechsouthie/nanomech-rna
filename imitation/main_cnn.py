import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt #type: ignore
from pathlib import Path
import numpy as np
import pandas as pd #type: ignore
from base_data import load_base_data
from base_CNN import BaseCNN, EternaDataset
from train_base_CNN import train_model, test_model
from datetime import datetime

NUM_FEATURES = 9
MAX_LENGTH = 350
NUM_CLASSES = 4
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
NUM_EPOCHS = 150
DROPOUT_RATE = 0.5  #dropout keep probability.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    print(f"Epochs: {NUM_EPOCHS}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print("Loading Data from the expert dataset folder...")

    project_root = Path(__file__).parent.parent  # imitation/main.py -> nanomech/
    data_folder = project_root / "X5"  # nanomech/X5
    
    print(f"Project root: {project_root}")
    print(f"Data folder: {data_folder}")
    print(f"Data folder exists: {data_folder.exists()}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_base_data(str(data_folder))

    if X_train is None:
        print("failed to load data.")
        return
    
    print(f"\nGot the data, data shapes:")
    print(f"    Train: X={X_train.shape}, y={y_train.shape}")
    print(f"    Val:  X={X_val.shape}, y={y_val.shape}")
    print(f"    Test: X={X_test.shape}, y={y_test.shape}")

    train_dataset = EternaDataset(X_train, y_train)
    val_dataset = EternaDataset(X_val, y_val)
    test_dataset = EternaDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=0, pin_memory=True)
    
    # Initialize model
    print(f"\n Creating model...")
    model = BaseCNN(num_features=NUM_FEATURES, seq_length=MAX_LENGTH, 
                   num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    # Loss, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=10, verbose=True)
    
    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Statistics:")
    print(f"  Architecture: 10 Conv blocks + 3 FC layers")
    print(f"  Activation: ReLU")
    print(f"  Normalization: BatchNorm")
    print(f"  Dropout rate: {DROPOUT_RATE}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max epochs: {NUM_EPOCHS}")
    
    # Train
    metrics = train_model(model, train_loader, val_loader, criterion, 
                         optimizer, scheduler, NUM_EPOCHS, DEVICE)
    
    # Plot and save metrics
    metrics.plot('training_metrics.png')
    metrics.save_csv('training_metrics.csv')
    
    # Load best model and test
    print("\n Loading best model for testing...")
    checkpoint = torch.load('models/base_CNN_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc = test_model(model, test_loader, DEVICE)
    
    # Save final model
    torch.save(model.state_dict(), 'models/base_CNN_final.pth')
    print("\n Final model saved to 'models/base_CNN_final.pth'")
    
    print("\n All done yayayay")


if __name__ == "__main__":
    main()