import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from base_data import load_base_data
from base_CNN import EternaDataset, MetricsTracker

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(inputs)

        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(output, 1)
        _, true_labels = torch.max(labels, 1)
        total += labels.size(0)
        correct += (predicted == true_labels).sum().item()

    epoch_loss = running_loss/ total
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, 
                num_epochs, device, checkpoint_dir='models'):
    print("-" * 60)
    print("starting training process")
    print("-" * 60)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    metrics = MetricsTracker()
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        metrics.update(epoch + 1, train_loss, val_loss, train_acc, val_acc)

        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'    Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%')
        print(f'    val loss:   {val_loss:.4f} | val acc:   {val_acc:.2f}%')
        print(f'    LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, f'{checkpoint_dir}/base_CNN_best.pth')

            print(f"Best model updated and saved. Best Val acc: {best_val_acc:.2f}%")
        else:
            patience_counter+= 1

        if patience_counter >= patience:
            print(f"Early stopping at {epoch+1} epochs due to no improvements after long no. of epochs")
            break

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'{checkpoint_dir}/base_CNN_epoch_{epoch+1}.pth')
            print("Checkpoint saved")

    print("\n"+"=" * 70)
    print("training complete")
    print(f"best validation accuracy: {best_val_acc:.2f}")



