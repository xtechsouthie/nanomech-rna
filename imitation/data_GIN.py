import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from typing import Tuple, Optional, List, Dict 
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import build_edge_index_attr, decode_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNAExpertDataset(Dataset):

    def __init__(self, data_folder: str, structure_file: Optional[str] = None, split: str = 'train',
                 max_length: int=400, test_size: float = 0.1, val_size: float = 0.1,
                random_state: int = 42, transform=None, pre_transform=None):
        
        super().__init__(None, transform, pre_transform)
        
        self.data_folder = data_folder
        self.structure_file = structure_file
        self.split = split
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self._load_base_data()
        self._create_split()

    def _get_file_map(self):
        logger.info(f"Scanning {self.data_folder} for data files...")

        try:
            x_files = glob(os.path.join(self.data_folder, "X5-exp-loc-*"))
            y_loc_files = glob(os.path.join(self.data_folder, "y5-exp-loc-*"))
            y_base_files = glob(os.path.join(self.data_folder, "y5-exp-base-*"))
        except Exception as e:
            logger.error(f"Error getting directory: {e}")
            print(f"Error getting the expert files: {e}")
            return {}, set()
        
        x_pids = {int(f.split("-")[-1]) for f in x_files}
        y_loc_pids = {int(f.split("-")[-1]) for f in y_loc_files}
        y_base_pids = {int(f.split("-")[-1]) for f in y_base_files}

        complete_pids = x_pids & y_loc_pids & y_base_pids
        print(f"Found {len(complete_pids)} complete puzzles.")

        file_map = {}
        for pid in complete_pids:
            file_map[pid] = {
                'features': os.path.join(self.data_folder, f"X5-exp-loc-{pid}"),
                'locations': os.path.join(self.data_folder, f"y5-exp-loc-{pid}"),
                'bases': os.path.join(self.data_folder, f"y5-exp-base-{pid}")
            }
        
        return file_map, complete_pids
    
    def _process_node_features(self, raw_features):

        seq_len = raw_features.shape[1]
        features = raw_features.copy()

        current_energy = features[3, 0] #broadcasting the current energy to all pos, abhi bass 1st pos pe hai
        target_energy = features[4, 0]

        features[3, :] = current_energy
        features[4, :] = target_energy

        node_features = torch.tensor(features.T, dtype=torch.float)

        return node_features

    def _load_pickle(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            return None
        
    def _load_base_data(self):

        file_map ,complete_pids = self._get_file_map()

        if not complete_pids:
            raise ValueError(f"No complete puzzles found in {self.data_folder}")
        
        self.all_features = []
        self.all_locations = []
        self.all_bases = []

        loaded_count = 0
        skipped_count = 0

        for pid in complete_pids:
            features = self._load_pickle(file_map[pid]['features'])
            locations = self._load_pickle(file_map[pid]['locations'])
            bases = self._load_pickle(file_map[pid]['bases'])

            if features is None or locations is None or bases is None:
                skipped_count += 1
                continue

            if len(features) != len(locations) or len(features) != len(bases):
                logger.warning(f"Length mismatch for puzzle {pid}, skipping")
                skipped_count += 1
                continue

            for i in range(len(features)):

                self.all_features.append(np.array(features[i]))
                self.all_locations.append(np.array(locations[i]))
                self.all_bases.append(np.array(bases[i]))

            loaded_count += 1

        logger.info(f"Loaded {loaded_count} puzzles, skipped {skipped_count}")
        logger.info(f"Total samples: {len(self.all_features)}")

    def _create_split(self):

        num_samples = len(self.all_features)
        indices = np.arange(num_samples)

        mutation_classes = np.array([np.argmax(b) for b in self.all_bases])

        train_val_idx, test_idx = train_test_split(
            indices, 
            test_size= self.test_size,
            random_state= self.random_state,
            stratify=mutation_classes
        )

        actual_val_size = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=actual_val_size,
            random_state=self.random_state,
            stratify=mutation_classes[train_val_idx]
        )

        if self.split == 'train': 
            self._indices = train_idx
        elif self.split == 'val': 
            self._indices = val_idx
        elif self.split == 'test': 
            self._indices = test_idx
        else:
            raise ValueError(f"unknown split: {self.split}")
        
        logger.info(f"split '{self.split}': {len(self._indices)} samples")

    def __len__(self):
        return len(self._indices)
    
    def indices(self):
        return self._indices
    
    def get(self, idx):


        raw_features = self.all_features[idx] #[8, seq_len]
        location_onehot = self.all_locations[idx] # [350]
        base_onehot = self.all_bases[idx] # [4]

        seq_length = len(raw_features[0])

        node_features = self._process_node_features(raw_features) #[ seq_len, 8]

        target_structure_tensor = torch.tensor(raw_features[2])
        current_structure = decode_structure(raw_features[1])

        edge_index_curr, edge_attr_curr = build_edge_index_attr(current_structure) #[2, E], [E, 4]

        location = int(np.argmax(location_onehot))
        mutation = int(np.argmax(base_onehot))

        data = Data(
            x = node_features,
            edge_attr=edge_attr_curr,
            edge_index=edge_index_curr,
            y_location = torch.tensor([location], dtype=torch.long),
            y_mutation = torch.tensor([mutation], dtype=torch.long),
            target_structure= target_structure_tensor,
            num_nodes = seq_length
        )

        return data



# def collate_fn(batch):
#     print("debug - collate function called")

#     batched = Batch.from_data_list(batch)

#     ptr = batched.ptr
#     local_locations = batched.y_location.squeeze(-1)

#     offsets = ptr[:-1]
#     global_locations = local_locations + offsets

#     batched.y_location_global = global_locations
#     batched.y_location_local = local_locations

#     return batched


def create_dataloaders(data_folder: str, batch_size: int = 64, test_size: float = 0.1,
        val_size: float = 0.1, num_workers: int = 0, random_state: int = 42):
    
    train_dataset = RNAExpertDataset(
        data_folder, split= 'train',
        test_size= test_size, val_size= val_size, random_state= random_state
    )
    val_dataset = RNAExpertDataset(
        data_folder, split= 'val',
        test_size = test_size, val_size=val_size, random_state= random_state
    )
    test_dataset = RNAExpertDataset(
        data_folder, split= 'test',
        test_size= test_size, val_size=val_size, random_state =random_state
    )

    
    train_loader = DataLoader(
        train_dataset,
        batch_size= batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle= False,
        num_workers= num_workers,
        pin_memory = True
    )
    
    return train_loader, val_loader, test_loader
    

def add_global_locations(batch):

    ptr = batch.ptr
    local_locations = batch.y_location.squeeze(-1)
    
    offsets = ptr[:-1]
    global_locations = local_locations + offsets
    
    batch.y_location_global = global_locations
    batch.y_location_local = local_locations
    
    return batch

  

