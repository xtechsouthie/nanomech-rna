import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from base_data import load_base_data

NUM_FEATURES = 9
MAX_LENGTH = 350
NUM_CLASSES = 4
BATCH_SIZE = 100
LEARNING_RATE = 0.0001
NUM_EPOCHS = 150
DROPOUT_RATE = 0.5  #dropout keep probability.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

class EternaDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
class BaseCNN(nn.Module):

    def __init__(self, num_features=NUM_FEATURES, seq_length=MAX_LENGTH, num_classes=NUM_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super(BaseCNN, self).__init__()

        self.num_features = num_features
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            self._make_conv_block(1, 2),
            self._make_conv_block(2, 4),
            self._make_conv_block(4, 8),
            self._make_conv_block(8, 16),
            self._make_conv_block(16, 32),
            self._make_conv_block(32, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
            self._make_conv_block(512, 1024),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),

            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),

            nn.Linear(4096, self.num_classes)
        )

        self._initialize_weights()

    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1) #yeh aaise kyu initialize kiya, for every module, conv, linear and batch norm
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = x.unsqueeze(1) # this adds the channel dimension, makes the 3d x into 4d
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)

        return x
    
class MetricsTracker:

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

    def plot(self, save_path="training_metrics.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

        ax1.plot(self.epochs, self.train_losses, label='Train Loss', marker='o', markersize=3)
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', marker='s', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.epochs, self.train_accs, label='Train Acc', marker='o', markersize=3)
        ax2.plot(self.epochs, self.val_accs, label='Val Acc', marker='s', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training metrics plot saved to {save_path}")

    def save_csv(self, save_path="training_metrics.csv"):
        df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs
        })
        df.to_csv(save_path, index=False)
        print(f"Training metircs saved to {save_path}")


