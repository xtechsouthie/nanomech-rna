# -*- coding: utf-8 -*-
"""
Extract Data from X5 Pickle Files (EternaBrain Pre-processed Data)
Created for: xtechsouthie
"""

import pickle
import numpy as np
import pandas as pd
import os
from glob import glob
import json

# ============================
# CONFIGURATION
# ============================

# Path to your X5 folder
X5_FOLDER = './X5/'

# Output directory
OUTPUT_DIR = './extracted_X5_data/'

# What to extract
EXTRACT_CONFIG = {
    'features': True,           # Extract X5 features
    'base_labels': True,        # Extract y5 base labels
    'location_labels': True,    # Extract y5 location labels
    'statistics': True,         # Calculate statistics
    'samples': True,            # Save sample data for inspection
}

# ============================
# HELPER FUNCTIONS
# ============================

def decode_base(one_hot):
    """
    Decode one-hot encoded base back to nucleotide
    [1,0,0,0] -> 'A'
    [0,1,0,0] -> 'U'
    [0,0,1,0] -> 'G'
    [0,0,0,1] -> 'C'
    """
    bases = ['A', 'U', 'G', 'C']
    idx = np.argmax(one_hot)
    return bases[idx]

def decode_location(one_hot):
    """
    Decode one-hot location to position index
    """
    return np.argmax(one_hot)

def decode_structure(encoded):
    """
    Decode structure: 1->'.' , 2->'(', 3->')'
    """
    structure_map = {1: '.', 2: '(', 3: ')'}
    return ''.join([structure_map. get(int(x), '.') for x in encoded])

def decode_sequence(encoded):
    """
    Decode sequence: 1->A, 2->U, 3->G, 4->C
    """
    seq_map = {1: 'A', 2: 'U', 3: 'G', 4: 'C'}
    return ''.join([seq_map.get(int(x), 'N') for x in encoded])

def load_pickle(filepath):
    """
    Load a pickle file safely
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None

# ============================
# FILE DISCOVERY
# ============================

def discover_files(folder):
    """
    Discover all X5 files and organize by puzzle ID
    """
    print(f"🔍 Discovering files in: {folder}")
    
    # Find all pickle files
    x_files = glob(os.path. join(folder, 'X5-exp-loc-*'))
    ybase_files = glob(os.path.join(folder, 'y5-exp-base-*'))
    yloc_files = glob(os. path.join(folder, 'y5-exp-loc-*'))
    
    # Extract puzzle IDs
    x_pids = [int(f.split('-')[-1]) for f in x_files]
    ybase_pids = [int(f.split('-')[-1]) for f in ybase_files]
    yloc_pids = [int(f.split('-')[-1]) for f in yloc_files]
    
    # Find puzzles with all three files
    complete_pids = set(x_pids) & set(ybase_pids) & set(yloc_pids)
    
    print(f"✅ Found:")
    print(f"   - {len(x_files)} X5-exp-loc files")
    print(f"   - {len(ybase_files)} y5-exp-base files")
    print(f"   - {len(yloc_files)} y5-exp-loc files")
    print(f"   - {len(complete_pids)} puzzles with complete data")
    
    # Organize by puzzle ID
    file_map = {}
    for pid in complete_pids:
        file_map[pid] = {
            'features': os.path.join(folder, f'X5-exp-loc-{pid}'),
            'base_labels': os.path.join(folder, f'y5-exp-base-{pid}'),
            'loc_labels': os.path.join(folder, f'y5-exp-loc-{pid}')
        }
    
    return file_map, complete_pids

# ============================
# DATA EXTRACTION
# ============================

def extract_features(file_map, pids):
    """
    Extract and analyze features from X5 files
    """
    print("\n📊 Extracting Features (X5-exp-loc files)...")
    
    feature_info = []
    
    for pid in pids:
        X = load_pickle(file_map[pid]['features'])
        if X is None:
            continue
        
        X = np.array(X)
        
        # Extract info
        info = {
            'pid': pid,
            'num_samples': len(X),
            'num_channels': X[0].shape[0] if len(X) > 0 else 0,
            'puzzle_length': X[0].shape[1] if len(X) > 0 else 0,
        }
        
        # Sample first move for this puzzle
        if len(X) > 0:
            sample = X[0]
            info['sample_sequence'] = decode_sequence(sample[0])
            info['sample_current_structure'] = decode_structure(sample[1])
            info['sample_target_structure'] = decode_structure(sample[2])
            info['sample_current_energy'] = float(sample[3][0])
            info['sample_target_energy'] = float(sample[4][0])
            info['sample_location'] = int(decode_location(sample[8]))
        
        feature_info.append(info)
    
    df = pd.DataFrame(feature_info)
    print(f"✅ Extracted features from {len(df)} puzzles")
    return df

def extract_base_labels(file_map, pids):
    """
    Extract base labels from y5-exp-base files
    """
    print("\n📊 Extracting Base Labels (y5-exp-base files)...")
    
    label_info = []
    
    for pid in pids:
        y = load_pickle(file_map[pid]['base_labels'])
        if y is None:
            continue
        
        y = np.array(y)
        
        # Decode labels
        bases = [decode_base(label) for label in y]
        
        info = {
            'pid': pid,
            'num_labels': len(y),
            'base_distribution': {
                'A': bases. count('A'),
                'U': bases.count('U'),
                'G': bases.count('G'),
                'C': bases.count('C')
            }
        }
        
        label_info.append(info)
    
    df = pd.DataFrame(label_info)
    print(f"✅ Extracted base labels from {len(df)} puzzles")
    return df

def extract_location_labels(file_map, pids):
    """
    Extract location labels from y5-exp-loc files
    """
    print("\n📊 Extracting Location Labels (y5-exp-loc files)...")
    
    location_info = []
    
    for pid in pids:
        y = load_pickle(file_map[pid]['loc_labels'])
        if y is None:
            continue
        
        y = np.array(y)
        
        # Decode positions
        positions = [decode_location(label) for label in y]
        
        info = {
            'pid': pid,
            'num_labels': len(y),
            'puzzle_length': y. shape[1] if len(y) > 0 else 0,
            'min_position': min(positions) if positions else 0,
            'max_position': max(positions) if positions else 0,
            'avg_position': np.mean(positions) if positions else 0
        }
        
        location_info.append(info)
    
    df = pd.DataFrame(location_info)
    print(f"✅ Extracted location labels from {len(df)} puzzles")
    return df

def extract_full_dataset(file_map, pid):
    """
    Extract complete dataset for a single puzzle
    """
    X = load_pickle(file_map[pid]['features'])
    y_base = load_pickle(file_map[pid]['base_labels'])
    y_loc = load_pickle(file_map[pid]['loc_labels'])
    
    if X is None or y_base is None or y_loc is None:
        return None
    
    X = np.array(X)
    y_base = np.array(y_base)
    y_loc = np.array(y_loc)
    
    dataset = []
    for i in range(len(X)):
        move = {
            'sample_idx': i,
            'current_sequence': decode_sequence(X[i][0]),
            'current_structure': decode_structure(X[i][1]),
            'target_structure': decode_structure(X[i][2]),
            'current_energy': float(X[i][3][0]),
            'target_energy': float(X[i][4][0]),
            'location': int(decode_location(X[i][8])),
            'target_base': decode_base(y_base[i]),
            'locks': X[i][7]. tolist()
        }
        dataset.append(move)
    
    return pd.DataFrame(dataset)

def calculate_statistics(file_map, pids):
    """
    Calculate overall statistics across all puzzles
    """
    print("\n📊 Calculating Overall Statistics...")
    
    total_samples = 0
    puzzle_lengths = []
    all_bases = []
    
    for pid in pids:
        X = load_pickle(file_map[pid]['features'])
        y_base = load_pickle(file_map[pid]['base_labels'])
        
        if X is not None and y_base is not None:
            X = np.array(X)
            y_base = np.array(y_base)
            
            total_samples += len(X)
            if len(X) > 0:
                puzzle_lengths.append(X[0].shape[1])
            
            bases = [decode_base(label) for label in y_base]
            all_bases.extend(bases)
    
    stats = {
        'total_puzzles': len(pids),
        'total_moves': total_samples,
        'avg_moves_per_puzzle': total_samples / len(pids) if len(pids) > 0 else 0,
        'puzzle_lengths': {
            'min': min(puzzle_lengths) if puzzle_lengths else 0,
            'max': max(puzzle_lengths) if puzzle_lengths else 0,
            'avg': np.mean(puzzle_lengths) if puzzle_lengths else 0
        },
        'base_distribution': {
            'A': all_bases.count('A'),
            'U': all_bases. count('U'),
            'G': all_bases.count('G'),
            'C': all_bases.count('C'),
            'A_pct': 100 * all_bases.count('A') / len(all_bases) if all_bases else 0,
            'U_pct': 100 * all_bases. count('U') / len(all_bases) if all_bases else 0,
            'G_pct': 100 * all_bases.count('G') / len(all_bases) if all_bases else 0,
            'C_pct': 100 * all_bases.count('C') / len(all_bases) if all_bases else 0,
        }
    }
    
    print(f"✅ Calculated statistics across {len(pids)} puzzles")
    return stats

# ============================
# MAIN PROCESSING
# ============================

def process_X5_data(folder, output_dir, extract_config):
    """
    Main function to process X5 pickle files
    """
    print("=" * 70)
    print("🧬 EXTRACTING DATA FROM X5 PICKLE FILES")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover files
    file_map, pids = discover_files(folder)
    
    if len(pids) == 0:
        print("❌ No complete puzzle datasets found!")
        return
    
    results = {}
    
    # Extract features
    if extract_config['features']:
        results['features'] = extract_features(file_map, pids)
        results['features']. to_csv(f"{output_dir}/feature_info.csv", index=False)
        print(f"💾 Saved: {output_dir}/feature_info.csv")
    
    # Extract base labels
    if extract_config['base_labels']:
        results['base_labels'] = extract_base_labels(file_map, pids)
        results['base_labels'].to_csv(f"{output_dir}/base_label_info.csv", index=False)
        print(f"💾 Saved: {output_dir}/base_label_info. csv")
    
    # Extract location labels
    if extract_config['location_labels']:
        results['location_labels'] = extract_location_labels(file_map, pids)
        results['location_labels'].to_csv(f"{output_dir}/location_label_info.csv", index=False)
        print(f"💾 Saved: {output_dir}/location_label_info. csv")
    
    # Calculate statistics
    if extract_config['statistics']:
        results['statistics'] = calculate_statistics(file_map, pids)
        with open(f"{output_dir}/statistics.json", 'w') as f:
            json.dump(results['statistics'], f, indent=2)
        print(f"💾 Saved: {output_dir}/statistics.json")
    
    # Extract sample datasets
    if extract_config['samples']:
        sample_pids = list(pids)[:5]  # First 5 puzzles as samples
        for pid in sample_pids:
            df = extract_full_dataset(file_map, pid)
            if df is not None:
                df.to_csv(f"{output_dir}/sample_puzzle_{pid}.csv", index=False)
                print(f"💾 Saved: {output_dir}/sample_puzzle_{pid}.csv")
    
    # Save puzzle list
    pd.DataFrame({'pid': list(pids)}).to_csv(f"{output_dir}/puzzle_ids.csv", index=False)
    print(f"💾 Saved: {output_dir}/puzzle_ids.csv")
    
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 70)
    
    if 'statistics' in results:
        stats = results['statistics']
        print(f"\n📊 Summary:")
        print(f"   - Total puzzles: {stats['total_puzzles']}")
        print(f"   - Total moves: {stats['total_moves']}")
        print(f"   - Avg moves/puzzle: {stats['avg_moves_per_puzzle']:.1f}")
        print(f"   - Puzzle length range: {stats['puzzle_lengths']['min']}-{stats['puzzle_lengths']['max']}")
        print(f"   - Base distribution:")
        print(f"     A: {stats['base_distribution']['A_pct']:.1f}%")
        print(f"     U: {stats['base_distribution']['U_pct']:.1f}%")
        print(f"     G: {stats['base_distribution']['G_pct']:.1f}%")
        print(f"     C: {stats['base_distribution']['C_pct']:.1f}%")
    
    return results, file_map, pids

# ============================
# RUN
# ============================

if __name__ == "__main__":
    results, file_map, pids = process_X5_data(
        folder=X5_FOLDER,
        output_dir=OUTPUT_DIR,
        extract_config=EXTRACT_CONFIG
    )
    
    # Print sample data
    print("\n" + "=" * 70)
    print("📋 SAMPLE DATA PREVIEW")
    print("=" * 70)
    
    if 'features' in results and results['features'] is not None:
        print("\n📊 FEATURE INFO (first 5 puzzles):")
        print(results['features'].head())