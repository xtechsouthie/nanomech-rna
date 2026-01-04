import argparse
import os
import sys
import logging
from copy import deepcopy
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch
import RNA
import yaml

from RNA_RL.actor_critic import ActorCritic
from utils import (get_graph, get_distance_from_base, get_energy_from_base,
    random_init_sequence_pair, update_graph_after_mutation, base_list
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DesignResult:
    target_structure: str
    designed_sequence: str
    folded_structure: str
    distance: int
    energy: float
    solved: bool

def get_pairmap(structure):

    pairs = [-1] * len(structure)
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            pairs[i], pairs[j] = j, i

    return pairs

def bracket_to_bonds(structure):
    bonds = []
    stack = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            bonds.append([j, i])

    return bonds

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def single_base_change(target, seq, iterations=4):

    target_pm = get_pairmap(target)
    seq = list(seq)
    bases = ['A', 'U', 'C', 'G']

    for _ in range(iterations):
        current = RNA.fold(''.join(seq))[0]
        if current == target:
            return ''.join(seq), True
        
        current_pm = get_pairmap(current)
        for loc in range(len(seq)):
            rewards = []
            for base in bases:
                test = seq.copy()
                test[loc] = base
                test_pm = get_pairmap(RNA.fold(''.join(test))[0])
                rewards.append(similar(test_pm, target_pm))

            if len(set(rewards)) > 1:
                seq[loc] = bases[np.argmax(rewards)]

    return ''.join(seq), RNA.fold(''.join(seq))[0] == target


def domain_specific_pipeline(target, seq):
    seq = list(seq)
    target_pm = get_pairmap(target)

    for i, pm in enumerate(target_pm):
        if pm == -1:
            seq[i] = 'A'
        elif pm > i:
            pair = (seq[i], seq[pm])
            valid = [('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')]
            if pair not in valid:
                seq[i], seq[pm] = 'G', 'C'

    # G booost
    for i in range(len(target) - 4):
        if target[i] == '(' and all(target[i+j] == '.' for j in range(1, 5)):
            seq[i+1] = 'G'

    
    best_seq = seq.copy()
    best_match = similar(get_pairmap(RNA.fold(''.join(seq))[0]), target_pm)

    for _ in range(3):
        for i, pm in enumerate(target_pm):
            if pm > i:
                seq[i], seq[pm] = seq[pm], seq[i]
                new_match = similar(get_pairmap(RNA.fold(''.join(seq))[0]), target_pm)
                if new_match > best_match:
                    best_match = new_match
                    best_seq = seq.copy()
                else:
                    seq[i], seq[pm] = seq[pm], seq[i]

                
    result = ''.join(best_seq)
    return result, RNA.fold(result)[0] == target

def refine_moves(target, seq, n_trajs = 50, n_steps = 20):
    valid_pairs = [['G','C'], ['C','G'], ['A','U'], ['U','A'], ['G','U'], ['U','G']]
    unpaired = [['A','A'], ['U','U'], ['C','C']]

    best_seq = seq
    best_acc = sum(1 for i in range(len(target)) if RNA.fold(seq)[0][i] == target[i]) / len(target)

    for _ in range(n_trajs):
        if best_acc >= 1.0:
            break

        current = list(best_seq)
        for _ in range(n_steps):
            pred = RNA.fold(''.join(current))[0]
            pred_bonds = bracket_to_bonds(pred)
            true_bonds = bracket_to_bonds(target)

            missing = [b for b in true_bonds if b not in pred_bonds]
            bad = [b for b in pred_bonds if b not in true_bonds]

            try:
                if missing and np.random.random() < 0.5:
                    pair = missing[np.random.randint(len(missing))]
                    mut = valid_pairs[np.random.randint(len(valid_pairs))]
                    current[pair[0]], current[pair[1]] = mut[0], mut[1]
                elif bad:
                    pair = bad[np.random.randint(len(bad))]
                    mut = unpaired[np.random.randint(len(unpaired))]
                    current[pair[0]], current[pair[1]] = mut[0], mut[1]
            except IndexError:
                continue

            new_acc = sum(1 for i in range(len(target)) if RNA.fold(''.join(current))[0][i] == target[i]) / len(target)
            if new_acc > best_acc:
                best_acc = new_acc
                best_seq = ''.join(current)

    return best_seq, RNA.fold(best_seq)[0] == target



class RNADesigner:
    def __init__(self, config):

        self.config = config
        self.device = torch.device(config['device'] if config['device'] != 'auto'
                                   else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"using device: {self.device}")

        self.model = ActorCritic(
            in_dim=config['network']['in_dim'],
            hidden_dim=config['network']['hidden_dim'],
            embed_dim=config['network']['embed_dim'],
            edge_dim=config['network']['edge_dim'],
            num_gnn_layers=config['network']['num_gnn_layers'],
            actor_hidden=config['network']['actor_hidden'],
            critic_hidden=config['network']['critic_hidden'],
            num_bases=config['network']['num_bases'],
            dropout=config['network']['dropout']
        ).to(self.device)

        checkpoint = torch.load(config['model_path'], map_location=self.device)
        state_dict = checkpoint.get('actor_critic', checkpoint.get('model_state_dict', checkpoint))

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"model loaded from path: {config['model_path']}")

    def design(self, target):
        cfg = self.config['inference']
        best_seq, best_dist = None, float('inf')

        for _ in range(cfg['num_samples']):
            init_seq, _ = random_init_sequence_pair(target, len(target))
            graph = get_graph(dotB=target, max_size=len(target), seq_base=init_seq)
            current_seq = init_seq

            for _ in range(cfg['max_steps']):
                if get_distance_from_base(current_seq, target) == 0:
                    break

                batch = Batch.from_data_list([graph])
                with torch.no_grad():
                    actions, _, _ = self.model(
                        x=batch.x.float().to(self.device),
                        edge_index=batch.edge_index.to(self.device),
                        edge_attr=batch.edge_attr.float().to(self.device),
                        batch=batch.batch.to(self.device),
                        ptr=batch.ptr.to(self.device),
                        deterministic=cfg['deterministic']
                    )

                loc, mut = actions.locations[0].item(), actions.mutations[0].item()
                seq_list = list(current_seq)
                seq_list[loc] = base_list[mut]
                current_seq = ''.join(seq_list)
                graph = update_graph_after_mutation(graph, current_seq, target)

            dist = get_distance_from_base(current_seq, target)
            if dist < best_dist:
                best_dist = dist
                best_seq = current_seq

        solved = (best_dist == 0)
        if cfg['use_post_processing'] and not solved:
            best_seq, solved = single_base_change(target, best_seq)
            if not solved:
                best_seq, solved = domain_specific_pipeline(target, best_seq)
            if not solved:
                best_seq, solved = refine_moves(target, best_seq, cfg['refine_trajs'], cfg['refine_steps'])

        folded = RNA.fold(best_seq)[0]

        return DesignResult(
            target_structure=target,
            designed_sequence=best_seq,
            folded_structure=folded,
            distance=get_distance_from_base(best_seq, target),
            energy=get_energy_from_base(best_seq, target),
            solved=(folded == target)
        )
    


def read_structures(path: str):
    structures = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and all(c in '.()' for c in line):
                structures.append(line)
    
    logger.info(f"loaded {len(structures)} structures from {path}")
    return structures

def write_results(path: str, results: List[DesignResult]):
    solved = sum(1 for r in results if r.solved)
    with open(path, 'w') as f:
        f.write(f"----> Solved {solved}/{len(results)} ({100*solved/len(results):.1f}%)\n\n")
        for i, r in enumerate(results):
            f.write(f"Puzzle {i+1}: {'SOLVED' if r.solved else f'dist={r.distance}'}\n")
            f.write(f"-- target: {r.target_structure}\n")
            f.write(f"-- SEQUENCE: {r.designed_sequence}\n")
            f.write(f"-- folded: {r.folded_structure}\n")
            f.write(f"-- energy: {r.energy:.2f} kcal/mol\n\n")

    logger.info(f"results written to {path}, solved: {solved}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(description="RNA sequence designeer")
    parser.add_argument("--config", type=str, default="config_inference.yaml", help="config file path")
    parser.add_argument("--model_path", type=str, help="override model path from config")
    parser.add_argument("--input", type=str, help="override input file from config")
    parser.add_argument("--output", type=str, help="override output file from config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.model_path:
        config['model_path'] = args.model_path
    if args.input:
        config['input_file'] = args.input
    if args.output:
        config['output_file'] = args.output
    
    if not os.path.exists(config['model_path']):
        logger.error(f"model not found: {config['model_path']}")
        sys.exit(1)
    if not os.path.exists(config['input_file']):
        logger.error(f"input file not found: {config['input_file']}")
        sys.exit(1)

    structures = read_structures(config['input_file'])
    designer = RNADesigner(config)

    results = []
    for i, structure in enumerate(structures):
        logger.info(f"designing {i+1}/{len(structures)} structure, length: {len(structure)}")
        results.append(designer.design(structure))

    write_results(config['output_file'], results)
    logger.info("\nRNA designing done...")

if __name__ == "__main__":
    main()


        