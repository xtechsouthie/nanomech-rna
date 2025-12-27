import gymnasium as gym
import numpy as np
from collections import namedtuple
import torch

from utils import (
    structure_dotB2Edge,
    random_init_sequence,
    random_init_sequence_pair,
    simple_init_sequence,
    get_graph,
    get_distance_from_base,
    get_energy_from_base,
    get_graph,
    rna_act,
    simplify_graph,
    seq_base2Onehot,
    onehot2base
)

Transition = namedtuple(
    'Transition',
    ['state', 'location', 'mutation', 'location_prob', 'mutation_prob', 'loc_reward',
     'mut_reward', 'next_state', 'done']
)

class RNA_design_env(gym.Env):
    """
    Gym environment for rna design problem

    State: graph representation of current RNA (sequence + structure)
    Action: (location, mutation) - which position to mutate and to which base
    Reward: based on structure distance and energy improvement
    """

    def __init__(
            self,
            target_structure=None,
            rna_id=None,
            init_seq=None,
            init_base_order=None,
            edge_threshold=0.001,
            max_size=None,
            action_space_size=4,
            reward_alpha=1.0,
            reward_beta=1.0,
            reward_gamma=1.0
    ):
        super(RNA_design_env, self).__init__()

        self.target_structure = target_structure
        self.rna_id = rna_id
        self.length = len(target_structure)
        self.max_size = max_size if max_size is not None else self.length

        self.edge_threshold = edge_threshold
        self.action_space_size = action_space_size
        self.base_list = ['A', 'U', 'C', 'G']

        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.reward_gamma = reward_gamma

        self.target_edge_index = structure_dotB2Edge(self.target_structure)    

        self.init_base_order = init_base_order

        if init_seq is None:
            if init_base_order is None:
                self.init_seq_base, self.init_seq_onehot = random_init_sequence_pair(self.target_structure, self.max_size, self.action_space_size)
            else:
                self.init_seq, self.init_seq_onehot = simple_init_sequence(
                    self.target_structure,
                    init_base_order,
                    self.max_size
                )
        else:
            self.init_seq_base = init_seq

        self.init_graph = get_graph(
            dotB=self.target_structure,
            max_size=self.max_size,
            seq_base=self.init_seq_base,
            edge_threshold=self.edge_threshold
        )

        self.current_graph = None
        self.current_seq = None
        self.last_distance = None
        self.last_energy = None
        self.done = False

        self.best_seq = self.init_seq_base
        self.best_distance = float('inf')

        self.step_count = 0
        self.max_steps = 1000

        self.reset()

    def reset(self):

        self.current_graph = self.init_graph.clone()
        self.current_seq = self.init_seq_base

        self.last_distance = self.current_graph.y['distance']
        self.last_energy = self.current_graph.y['real_energy']

        self.done = (self.last_distance == 0)

        self.step_count = 0

        if self.last_distance < self.best_distance:
            self.best_seq = self.current_seq
            self.best_distance = self.last_distance

        return self.current_graph
    
    def step(self, action): # action ko tuple daldiya
        
        location, mutation = action

        if location < 0 or location > self.length:
            print(f"Invalid location {location}")

        new_graph = rna_act(
            self.current_graph, 
            location=location,
            mutation=mutation,
            action_space=self.action_space_size
        )

        new_seq = new_graph.y['seq_base']
        new_distance = new_graph.y['distance']
        new_energy = new_graph.y['real_energy']

        loc_reward, mut_reward = self.compute_seprated_rewards(
            location, mutation, new_distance, new_energy
        )

        self.current_graph = new_graph
        self.current_seq = new_seq
        self.last_distance = new_distance
        self.last_energy = new_energy

        self.step_count += 1

        self.done = (new_distance == 0) or (self.step_count >= self.max_steps)

        if new_distance <= self.best_distance:
            self.best_seq = new_seq
            self.best_distance = new_distance

        if new_distance == 0:
            loc_reward += 5.0
            mut_reward += 5.0

        info = {
            'distance': new_distance,
            'energy': new_energy,
            'sequence': new_seq,
            'step': self.step_count,
            'best_distance': self.best_distance,
            'structure': new_graph.y['real_dotB']
        }

        return new_graph, (loc_reward, mut_reward), self.done, info
    
    def compute_seprated_rewards(self, location, mutation, new_distance, new_energy):

        location_reward = self._evaluate_location_quality(location)

        distance_improvement = self.last_distance - new_distance

        energy_improvement = self.last_energy - new_energy

        whole_reward = ((self.reward_alpha * distance_improvement) + (self.reward_beta * energy_improvement))

        mutation_reward = whole_reward

        location_reward = 0.7 * location_reward + 0.3 * whole_reward

        return location_reward, mutation_reward
    
    def _evaluate_location_quality(self, location):

        candidate_improvements = []

        for mut_idx in range(self.action_space_size):

            if self.base_list[mut_idx] == self.current_seq[location]:
                continue

            candidate_graph = rna_act(
                self.current_graph,
                location=location,
                mutation=mut_idx,
            )

            candidate_distance = candidate_graph.y['distance']
            candidate_energy = candidate_graph.y['real_energy']

            distance_improvement = self.last_distance - candidate_distance
            energy_improvement = self.last_energy - candidate_energy

            improvement = (self.reward_alpha * distance_improvement +
                          self.reward_beta * energy_improvement)

            candidate_improvements.append(improvement)

        if len(candidate_improvements) == 0:
            return 0.0

        max_improvement = max(candidate_improvements)
        avg_improvement = np.mean(candidate_improvements)

        location_reward = 0.7 * max_improvement + 0.3 * avg_improvement

        return location_reward

    def _evaluate_mutation_quality(self, location, mutation):

        new_seq = self._apply_mutation(self.current_seq, location, mutation)

        new_distance = get_distance_from_base(new_seq, self.target_structure)
        new_energy = get_energy_from_base(new_seq, self.target_structure)

        distance_improvement = self.last_distance - new_distance
        energy_improvement = self.last_energy - new_energy

        mutation_reward = (self.reward_alpha * distance_improvement +
                          self.reward_beta * energy_improvement)
        
        return mutation_reward
    
    def _apply_mutation(self, sequence, location, mutation):

        seq_list = list(sequence)
        seq_list[location] = self.base_list[mutation]
        return ''.join(seq_list)
    
    def get_single_indices(self):

        single_indices = [i for i, char in enumerate(self.target_structure) if char == "."]
        return single_indices
    
    def render(self, mode='human'):

        if mode == 'human':
            print(f"\n{'-'*60}")
            print(f"rna design environment - ID: {self.rna_id}")
            print(f"step: {self.step_count}")
            print(f"Target structure: {self.target_structure}")
            print(f"Current sequence: {self.current_seq}")
            print(f"distance: {self.last_distance}")
            print(f"energy:  {self.last_energy:. 2f}")
            print(f"Best distance: {self.best_distance}")
            print(f"done: {self.done}")
            print(f"{'-'*60}\n")

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.random.seed(seed)

    def __len__(self):
        return self.length
    
    def get_state(self):
        return self.current_graph
    
    def get_info(self):

        return {
            'rna_id': self.rna_id,
            'sequence': self.current_seq,
            'distance': self.last_distance,
            'energy': self.last_energy,
            'step': self.step_count,
            'done': self.done,
            'best_sequence': self.best_seq,
            'best_distance': self. best_distance
        }
