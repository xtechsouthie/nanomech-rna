import os
from glob import glob
import pickle 
import pandas as pd #type: ignore
import numpy as np
from sklearn.model_selection import train_test_split #type: ignore

MAX_LENGTH = 350 #maximum length of the puzzles

def get_data(folder: str):
    print("Getting Data...")

    try:
        X_files = glob(os.path.join(folder, "X5-exp-loc-*"))
        y_loc_files = glob(os.path.join(folder, "y5-exp-loc-*"))
    except Exception as e:
        print(f"error getting files: {e}")

    X_pids = [int(f.split("-")[-1]) for f in X_files]
    y_loc_pids = [int(f.split("-")[-1]) for f in y_loc_files]

    complete_pids = set(X_pids) & set(y_loc_pids)

    file_map = {}
    for pid in complete_pids:
        file_map[pid] = {
            "features": os.path.join(folder, f"X5-exp-loc-{pid}"),
            "loc_labels": os.path.join(folder, f"y5-exp-loc-{pid}")
        }
    
    return file_map, complete_pids

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading file {file}:  {e}")
        return None

def load_data(file_map, complete_pids):
    print("Loading Data for loc CNN...")

    full_X = []
    full_y_loc = []
    drop_pids = []

    for pid in complete_pids:
        feats_file = file_map[pid]["features"]
        yloc_file = file_map[pid]["loc_labels"]

        feats = load_pickle(feats_file)
        yloc = load_pickle(yloc_file)

        if feats is None or yloc is None:
            print(f"Failed to load files for pid: {pid}")
            drop_pids.append(pid)
            continue

        if len(feats) != len(yloc):
            print(f"Length mismatch in yloc pid: {pid} (feats={len(feats)}, yloc={len(yloc)})")
            drop_pids.append(pid)
            continue

        full_X.extend(feats)
        full_y_loc.extend(yloc)

    for pid in drop_pids:
        if pid in complete_pids:
            complete_pids.remove(pid)

    print(f"Dropped {len(drop_pids)} puzzles due to errors")

    return full_X, full_y_loc

def load_loc_data(folder: str):

    file_map, complete_pids = get_data(folder)
    full_X, full_y = load_data(file_map, complete_pids)

    puzzle_length = []
    for puz in range(len(full_X)):
        length = len(full_X[puz][0])
        puzzle_length.append(length)

    max_length = max(puzzle_length)
    
    for i, datapoint in enumerate(full_X):
        if len(datapoint[0]) < max_length:
            k = len(datapoint[0])
            for j in range(len(datapoint)):
                datapoint[j].extend([0] * (max_length - k))

    X = np.array(full_X)
    y = np.array(full_y)

    y_classes = np.argmax(y, axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size = 0.2,
        random_state = 69,
        stratify = y_classes
    )

    y_temp_classes = np.argmax(y_temp, axis=1)

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=69,
        stratify=y_temp_classes
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
