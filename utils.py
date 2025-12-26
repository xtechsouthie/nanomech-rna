import RNA
import numpy as np
import torch
import pickle
from distance import hamming
from typing import Optional
import random
from torch_geometric.data import Data

base_list = ['A', 'U', 'C', 'G']
base_to_idx = {'A': 1, 'U': 2, 'C': 3, 'G': 4}
structure_to_idx = {'.': 1, '(': 2, ')': 3}
onehot_list = [np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])]
base_pair_list_4 = [['A', 'U'], ['U', 'A'], ['C', 'G'], ['G', 'C']]
base_pair_list_6 = [['A', 'U'], ['U', 'A'], ['U', 'G'], ['G', 'U'], ['G', 'C'], ['C', 'G']]

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

def get_pairing_partner(dotB: str):
    l = len(dotB)

    partners = np.full(l, -1, dtype=np.float32)
    stack = []

    for i, char in enumerate(dotB):
        if char == "(":
            stack.append(i)
        elif char == ")":
            j = stack.pop()
            partners[i] = j
            partners[j] = i

    return partners

def get_pair_indices_from_dotB(dotB: str):
    pairs = []
    stack = []

    for i, char in enumerate(dotB):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            pairs.append((j, i))

    return pairs

def encode_structure(dotB: str):
    return np.array([structure_to_idx.get(char, 1) for char in dotB])

def encode_sequence(seq_base: str):
    return np.array([base_to_idx.get(base, 1) for base in seq_base]) 

def decode_structure(encoded: np.ndarray):

    structure_map = {1: '.', 2: '(', 3: ')'}
    return ''.join([structure_map.get(int(x), '.') for x in encoded])

def load_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return None


def build_edge_index_attr(structure, seq_length=None): #give dotB structure

    if seq_length is None:
        seq_length = len(structure)

    edge_index = []
    edge_attrs = []

    for i in range(seq_length - 1):
        edge_index.append([i, i+1])
        edge_attrs.append([1.0, 0.0, 1.0, 1.0]) # this is [is_backbone, is_pair, weight, distance]
        edge_index.append([i + 1, i])
        edge_attrs.append([1.0, 0.0, 1.0, 1.0])

    stack = []
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                edge_index.append([j, i])
                distance = abs(i - j)
                edge_attrs.append([0.0, 1.0, 1.0, float(distance)])
                edge_index.append([i, j])
                edge_attrs.append([0.0, 1.0, 1.0, float(distance)])

    if len(edge_index) == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 4), dtype=torch.float)
    
    edge_index = torch.LongTensor(edge_index).T # [2, num_edges]
    edge_attr = torch.FloatTensor(edge_attrs) # [num_edges, 4]

    return edge_index, edge_attr
    
def collate_graphs(batch):

    batch_size = len(batch)

    node_features_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []
    locations = []
    mutations = []
    seq_lengths = []

    node_offset = 0
    ptr = [0]

    for i, sample in enumerate(batch):
        num_nodes = sample['node_features'].size(0)

        node_features_list.append(sample['node_features'])

        edge_index = sample['edge_index'] + node_offset

        edge_attr_list.append(sample(['edge_attr']))
        edge_index_list.append(edge_index)

        batch_list.append(torch.full((num_nodes, ), i, dtype=torch.long))

        locations.append(sample['location'] + node_offset)  # Offset location too
        mutations.append(sample['mutation'])
        seq_lengths.append(sample['seq_length'])

        node_offset += num_nodes
        ptr.append(node_offset)

    return {
        'node_features':  torch.cat(node_features_list, dim=0),  # [total_nodes, feat_dim]
        'edge_index': torch.cat(edge_index_list, dim=1),      # [2, total_edges]
        'edge_attr': torch.cat(edge_attr_list, dim=0),     # [total_edges, edge_dim]
        'batch': torch.cat(batch_list, dim=0),        # [total_nodes]
        'locations': torch.stack(locations),        # [batch_size]
        'mutations': torch.stack(mutations),     # [batch_size]
        'seq_lengths':  torch.stack(seq_lengths),    # [batch_size]
        'ptr': torch.LongTensor(ptr)    # [batch_size + 1]
    }


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

def build_node_features_8ch(seq_base: str, current_structure: str, 
                            target_structure: str, current_energy: float, target_energy: float):
    
    seq_len = len(seq_base)

    ch0 = encode_sequence(seq_base)
    ch1 = encode_structure(current_structure)
    ch2 = encode_structure(target_structure)
    ch3 = np.full(seq_len, current_energy, dtype=np.float32)
    ch4 = np.full(seq_len, target_energy, dtype=np.float32)
    ch5 = get_pairing_partner(current_structure)
    ch6 = get_pairing_partner(target_structure)
    ch7 = np.ones(seq_len, dtype=np.float32)

    features = np.stack([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7], axis=0)
    features = torch.Tensor(features.T, dtype=torch.float32)

    return features    

def random_base_pair(action_space: int):
    pair_list = base_pair_list_4 if action_space == 4 else base_pair_list_6
    idx = random.randrange(0, len(pair_list))
    return pair_list[idx][0], pair_list[idx][1]


def random_init_sequence_pair(dotB: str, max_size: int, action_space: int = 4):

    l = len(dotB)
    seq_list = [''] * l

    pairs = get_pair_indices_from_dotB(dotB)
    paired_positions = set()

    for i, j in pairs:
        base_i, base_j = random_base_pair(action_space)
        seq_list[i] = base_i
        seq_list[j] = base_j
        paired_positions.add(i)
        paired_positions.add(j)

    for i in range(l):
        if i not in paired_positions:
            seq_list[i], _ = random_base()

    seq_base = ''.join(seq_list)
    seq_onehot = seq_base2Onehot(seq_base, max_size)

    return seq_base, seq_onehot


def get_graph(dotB=None, max_size=None, seq_base=None, seq_onehot=None, edge_threshold=0.001):
        
    if max_size is None:
        max_size = len(dotB)

    seq_length = len(dotB)
    
    if seq_base is None:
        seq_base, seq_onehot = random_init_sequence_pair(dotB, max_size)
    elif seq_onehot is None:
        seq_onehot = seq_base2Onehot(seq_base, max_size)

    real_structure, real_energy = RNA.fold(seq_base)

    target_energy = get_energy_from_base(seq_base, dotB)

    node_features = build_node_features_8ch(
        seq_base = seq_base,
        current_structure= real_structure,
        target_structure= dotB,
        current_energy= real_energy,
        target_energy= target_energy
    )

    aim_edge_index = structure_dotB2Edge(dotB)
    real_edge_index = structure_dotB2Edge(real_structure)

    edge_index, edge_attr = build_edge_index_attr(real_structure, seq_length)

    distance = get_distance_from_base(seq_base, dotB)

    single_indices = [i for i, char in enumerate(dotB) if char == "."]
    pair_indices = [i for i, char in enumerate(dotB) if char == "("]

    y_dict = {
        'dotB': dotB,
        'seq_base': seq_base,
        'real_dotB': real_structure,
        'aim_energy': target_energy,
        'real_energy': real_energy,
        'distance': distance,
        'single_index': single_indices,
        'pair_index': pair_indices,
        'aim_edge_index': aim_edge_index,
        'real_edge_index': real_edge_index,
        'length': seq_length
    }

    graph = Data(
        x = node_features,
        edge_index= edge_index,
        edge_attr = edge_attr,
        y= y_dict
    )

    return graph

def update_graph_after_mutation(graph: Data, new_seq: str, target_dotB: str):
    seq_length = len(new_seq)

    real_structure, real_energy = RNA.fold(new_seq)
    target_energy = get_energy_from_base(new_seq, target_dotB)

    node_features = build_node_features_8ch(
        seq_base= new_seq,
        current_structure= real_structure, 
        target_structure= target_dotB,
        current_energy= real_energy,
        target_energy= target_energy
    )

    edge_index, edge_attr = build_edge_index_attr(real_structure, seq_length)

    distance = get_distance_from_base(new_seq, target_dotB)

    real_edge_index = structure_dotB2Edge(real_structure)
    
    graph.x = node_features
    graph.edge_index = edge_index
    graph.edge_attr = edge_attr

    graph.y['seq_base'] = new_seq
    graph.y['real_dotB'] = real_structure
    graph.y['aim_energy'] = target_energy
    graph.y['real_energy'] = real_energy
    graph.y['distance'] = distance
    graph.y['real_edge_index'] = real_edge_index

    return graph


def rna_act(graph: Data, location: int, mutation: int, action_space: int = 4):

    seq_list = list(graph.y['seq_base'])
    target_dotB = graph.y['dotB']

    seq_list[location] = base_list[mutation]

    new_seq = ''.join(seq_list)

    graph = update_graph_after_mutation(graph, new_seq, target_dotB)

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