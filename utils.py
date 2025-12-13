import RNA
import numpy as np
import torch
from distance import hamming
import random
from torch_geometric.data import Data

base_list = ['A', 'U', 'C', 'G']
onehot_list = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]


def structure_dotB2Edge(dotB):
    l = len(dotB)
    u = []
    v = []

    for i in range(l-1):
        u += [i, i+1]
        v += [i+1, i]
    str_list = list(dotB)
    stack = []
    for i in range(l):
        if (str_list[i] == "("):
            stack.append(i)
        elif (str_list[i] == ")"):
            last_place = stack.pop()
            u += [i, last_place]
            v += [last_place, i]
    edge_index = torch.tensor(np.array([u, v]))
    return edge_index

def base2Onehot(base):
    onehot = torch.tensor(np.zeros((4,)), dtype=torch.long)
    i = 0
    if (base == "A"):
        i = 0
    elif (base == "U"):
        i = 1
    elif (base == "C"):
        i = 2
    else:
        i = 3
    onehot[i] = 1

    return onehot

def seq_base2Onehot(seq_base, max_size=None):
    
    l = len(seq_base)
    seq_onehot = map(base2Onehot, list(seq_base))
    seq_onehot = list(seq_onehot)
    if max_size is not None:
        seq_onehot += [torch.tensor([0, 0, 0, 0]) for i in range(max_size - l)]
    seq_onehot = torch.stack(seq_onehot, dim=0)
    return seq_onehot

def get_energy_from_base(seq_base, dotB):
    energy = RNA.energy_of_struct(seq_base, dotB)
    return energy

def get_distance_from_base(seq_base, dotB_Aim):
    dotB_Real = RNA.fold(seq_base)[0]
    distance = hamming(dotB_Real, dotB_Aim)
    return distance

def random_base():
    one_hot = [0, 0, 0, 0]
    seed = random.randrange(0, 4, 1)
    base = base_list[seed]
    one_hot[seed] = 1
    return base, one_hot

def random_init_sequence(dotB, max_size=None):
    l = len(dotB)
    seq_base = ""
    seq_onehot = []

    for i in range(l):
        base_temp, onehot_temp = random_base()
        seq_base += base_temp
        seq_onehot.append(onehot_temp)
    
    if max_size is not None:
        seq_onehot += [[0, 0, 0, 0] for i in range(max_size - l)]

    seq_onehot = torch.tensor(seq_onehot)
    return seq_base, seq_onehot

def simple_init_sequence(dotB, base_order, max_size=None):
    l = len(dotB)
    seq_base = '.'.join([base_list[base_order] for i in range(l)])
    seq_onehot = [onehot_list[base_order] for i in range(l)]
    if max_size is not None:
        seq_onehot += [[0, 0, 0, 0] for i in range(max_size - l)]
    seq_onehot = torch.tensor(seq_onehot)
    return seq_base, seq_onehot


def get_graph(dotB=None, max_size=None, seq_base=None, seq_onehot=None, edge_threshold=0.001):
        
    if max_size is None:
        max_size = len(seq_base if seq_base is not None else dotB)
    if seq_base == None:
        seq_base, seq_onehot = random_init_sequence(dotB, max_size)
    else:
        if seq_onehot is None:
            seq_onehot = seq_base2Onehot(seq_base)

    seq_length = len(dotB)
    node_features = seq_onehot

    edge_index = structure_dotB2Edge(dotB)

    num_edges = edge_index.size(1)
    edge_attr = torch.ones((num_edges, 1), dtype=torch.float)

    real_structure, real_energy = RNA.fold(seq_base)
    distance = get_distance_from_base(seq_base, dotB)

    single_indices = [i for i, char in enumerate(dotB) if char == "."]

    stack = []
    pair_indices = []
    for i, char in enumerate(dotB):
        if char == "(":
            pair_indices.append(i)
            stack.append(i)
        elif char == ")":
            h = stack.pop()

    aim_adj = torch.zeros((seq_length, seq_length), dtype=torch.float)
    real_adj = torch.zeros((seq_length, seq_length), dtype=torch.float)

    stack = []
    for i, char in enumerate(dotB):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                aim_adj[i, j] = 1
                aim_adj[j, i] = 1
    
    stack = []
    for i, char in enumerate(real_structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                real_adj[i, j] = 1
                real_adj[j, i] = 1

    y_dict = {
        'dotB': dotB,
        'seq_base': seq_base,
        'real_dotB': real_structure,
        'energy': real_energy,
        'distance': distance,
        'single_index': torch.tensor(single_indices, dtype=torch.long),
        'pair_index': torch.tensor(pair_indices, dtype=torch.long),
        'aim_adj': aim_adj,
        'real_adj': real_adj,
        'length': seq_length,
    }

    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr, # could make it more informative like edge_attr.append([is_backbone, is_pair, 1.0, dist])
        y=y_dict
    )

    return graph

def simplify_graph(graph):
    simplified = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        y={
            'dotB': graph.y['dotB'],
            'seq_base': graph.y['seq_base'],
            'real_dotB': graph.y['real_dotB'],
            'energy': graph.y['energy'],
            'distance': graph.y['distance'],
            'length': graph.y['length'],
        }
    )
    return simplified


def onehot2base(onehot):
    if isinstance(onehot, torch.Tensor):
        idx = torch.argmax(onehot).item()
    else:
        idx = np.argmax(onehot)
    
    return base_list[idx]