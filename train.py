import os
import sys
import logging
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import yaml
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RNA_RL.environment import RNA_design_env
from RNA_RL.actor_critic import ActorCritic
from RNA_RL.ppo import PPO, RolloutBuffer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config_rl.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def curriculum_sample_structures(structures, max_structures, bin_size, min_bin_count):

    bins = defaultdict(list)
    for rna_id, structure in structures:
        length = len(structure)
        bin_idx = length // bin_size
        bins[bin_idx].append((rna_id, structure))

    valid_bins = {k: v for k, v in bins.items() if len(v) >= min_bin_count}

    if not valid_bins:
        logger.warning(f"no bins with >= {min_bin_count} structures found, using all structures")
        sorted_structures = sorted(structures, key=lambda x: len(x[1]))
        return sorted_structures[:max_structures]
    
    num_bins = len(valid_bins)
    per_bin = max_structures // num_bins

    sampled = []
    for bin_idx in sorted(valid_bins.keys()):
        bin_structures = valid_bins[bin_idx]
        n_sample = min(per_bin, len(bin_structures))
        sampled.extend(random.sample(bin_structures, n_sample))

    sampled.sort(key=lambda x: len(x[1]))

    logger.info(f"curriculum learning samples: {len(sampled)} structures from {num_bins} bins")
    for bin_idx in sorted(valid_bins.keys()):
        bin_lens = [len(s) for _, s in valid_bins[bin_idx]]
        logger.info(f"bin {bin_idx} [{min(bin_lens)}-{max(bin_lens)}]: {len(valid_bins[bin_idx])} structures")
    
    return sampled

def load_structures_from_rna_files(data_dir="../data_rl/rfam_learn/train", max_structures=None, use_curriculum = True, bin_size=50, min_bin_count=150):

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    
    structures = []

    rna_files = sorted(data_path.glob('*.rna'), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)

    for rna_file in rna_files:
        try:
            rna_id = int(rna_file.stem) if rna_file.stem.isdigit() else hash(rna_file.stem) % 100000

            with open(rna_file, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue

                structure = None
                for line in reversed(lines):
                    line = line.strip()
                    if line and all(c in '.()' for c in line):
                        structure = line
                        break
                
                count = 0
                valid = True
                for c in structure:
                    if c == '(':
                        count += 1
                    elif c == ')':
                        count -= 1
                    
                    if count < 0:
                        valid = False
                        break

                if valid and count == 0:
                    structures.append((rna_id, structure))

        except Exception as e:
            continue

    if max_structures and len(structures) > max_structures:
        if use_curriculum:
            structures = curriculum_sample_structures(structures, max_structures, bin_size, min_bin_count)
        else:
            structures = random.sample(structures, max_structures)

    elif use_curriculum:
        structures.sort(key=lambda x: len(x[1]))

    logger.info(f"loaded {len(structures)} structures from the data dir {data_dir}")
    return structures


def collect_rollout(env: RNA_design_env, ppo: PPO, buffer: RolloutBuffer, max_steps: int):

    state = env.reset()
    ep_ret, ep_len = 0, 0

    for _ in range(max_steps):

        loc, mut, loc_lp, mut_lp, loc_v, mut_v = ppo.select_action(state)

        next_state, reward, done, info = env.step((loc, mut))

        loc_reward, mut_reward = reward
        total_reward = 0.5 * (loc_reward + mut_reward)

        ep_ret += total_reward
        ep_len += 1

        buffer.store(
            state=state,
            location= loc,
            mutation= mut,
            loc_log_prob=loc_lp,
            mut_log_prob= mut_lp,
            loc_value=loc_v,
            mut_value=mut_v,
            loc_reward=loc_reward,
            mut_reward=mut_reward,
            done= done
        )

        state = next_state
        if done:
            break

    if done:
        last_loc_v, last_mut_v = 0, 0
    else:
        last_loc_v, last_mut_v = ppo.get_value(state)

    buffer.finish_path(last_loc_val=last_loc_v, last_mut_val=last_mut_v)

    return {
        'ep_ret': ep_ret,
        'ep_len': ep_len,
        'distance': info.get('distance', -1),
        'solved': info.get('distance', -1) == 0,
        'rna_id': getattr(env, 'rna_id', 0)
    }

def evaluate(ppo, test_structure, max_ep_len, num_samples=10):
    logger.info(f"evaluating on {len(test_structure)} test structures..")

    if num_samples and len(test_structure) > num_samples:
        eval_structures = random.sample(test_structure, num_samples)
        logger.info(f"evaluating on {num_samples} sampled structures (out of {len(test_structure)})")
    else:
        eval_structures = test_structure
        logger.info(f"evaluating on {len(eval_structures)} test structures...")

    ep_rets, distances = [], []
    solved_count = 0

    for rna_id, structure in tqdm(test_structure, desc="evaluating"):
        try:
            env = RNA_design_env(target_structure=structure)
            env.rna_id = rna_id
            env.max_steps = max_ep_len

            state = env.reset()
            ep_ret = 0

            for _ in range(max_ep_len):
                with torch.no_grad():
                    loc, mut, _, _, _, _ = ppo.select_action(state)

                state, reward, done, info = env.step((loc, mut))
                loc_reward, mut_reward = reward
                ep_ret += 0.5 * (loc_reward + mut_reward)

                if done:
                    break

            ep_rets.append(ep_ret)
            distances.append(info.get('distance', -1))
            if info.get('distance', -1) == 0:
                solved_count += 1
        
        except Exception as e:
            logger.warning(f"error evaluating rna id: {rna_id}: {e}")


    return {
        'avg_return': np.mean(ep_rets) if ep_rets else 0,
        'avg_distance': np.mean(distances) if distances else 0,
        'solve_rate': solved_count / len(test_structure) if test_structure else 0,
        'solved_count': solved_count
    }

def train(config_path: str = 'config_rl.yaml'):

    config = load_config(config_path)

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if config['device']['use_cuda'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(config['checkpoint']['save_dir']) / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / 'config_run.yaml', 'w') as f:
        yaml.dump(config, f)

    train_structures = load_structures_from_rna_files(
        config['data']['data_dir'], 
        config['data'].get('max_structures', None),
        use_curriculum=config['data'].get('use_curriculum', True),
        bin_size=config['data'].get('bin_size', 50),
        min_bin_count=config['data'].get('min_bin_count', 150)
    )

    if not train_structures:
        logger.error("structure loading failed successfully wow")
        return

    test_structures = load_structures_from_rna_files(
        config['data']['val_dir'], config['data'].get('max_structures', None), use_curriculum=False
    )

    if not test_structures:
        logger.info("test structures failed to load")

    actor_critic = ActorCritic(
        in_dim=config['network']['in_dim'],
        hidden_dim=config['network']['hidden_dim'],
        embed_dim=config['network']['embed_dim'],
        edge_dim=config['network']['edge_dim'],
        num_gnn_layers=config['network']['num_gnn_layers'],
        actor_hidden=config['network']['actor_hidden'],
        critic_hidden=config['network']['critic_hidden'],
        num_bases=config['network']['num_bases'],
        dropout=config['network']['dropout']
    ).to(device)

    pretrained_path = config['training'].get('pretrained', None)
    if pretrained_path and os.path.exists(pretrained_path):
        pretrain_info = actor_critic.load_pretrained_bc(
            pretrained_path,
            load_backbone=True,
            load_actors=True,
            freeze_backbone=config['training'].get('freeze_backbone_epochs', 0) > 0,
            freeze_actors=config['training'].get('freeze_actors_epochs', 0) > 0
        )
        logger.info(f"loaded pretrained weigths from {pretrained_path}")
        logger.info(f"BC checkpoint info: {pretrain_info.get('bc_info', {})}")
    else:
        logger.warning("no pretrained weights loaded, training from scratch")


    ppo = PPO(
        actor_critic=actor_critic,
        lr_actor=config['training']['lr_actor'],
        lr_critic=config['training']['lr_critic'],
        lr_backbone=config['training']['lr_backbone'],
        gamma=config['ppo']['gamma'],
        lam=config['ppo']['lam'],
        clip_ratio=config['ppo']['clip_ratio'],
        target_kl=config['ppo']['target_kl'],
        entropy_coef=config['ppo']['entropy_coef'],
        train_actor_iters=config['ppo']['train_actor_iters'],
        train_critic_iters=config['ppo']['train_critic_iters'],
        device=device
    )

    buffer = RolloutBuffer(
        size=config['training']['steps_per_epoch'],
        gamma= config['ppo']['gamma'],
        lam=config['ppo']['lam']
    )

    start_time = time.time()
    best_solve_rate = 0
    solved_set = set()

    logger.info("=" * 60)
    logger.info("\n starting PPO training")
    logger.info(f"train structures: {len(train_structures)} | test structures: {len(test_structures)}")
    logger.info(f"epochs: {config['training']['num_epochs']}")
    logger.info("=" * 60)

    global_structure_idx = 0

    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()

        if epoch == config['training'].get('freeze_backbone_epochs', 0):
            actor_critic.unfreeze_backbone()
            logger.info(f"epoch {epoch}: unfreezing the backbone")

        if epoch == config['training'].get('freeze_actor_epoch', 0):
            actor_critic.unfreeze_all()
            logger.info(f"epoch {epoch}: unfreexing the actors")

        # shuffled_structures = train_structures.copy()
        # random.shuffle(shuffled_structures)
        
        ep_rets, ep_lens, distances = [], [], []
        solved_count = 0
        steps = 0

        epoch_start_idx = global_structure_idx
        
        with tqdm(total=config['training']['steps_per_epoch'], desc=f'Epoch {epoch+1}') as pbar:
            while steps < config['training']['steps_per_epoch']:
                rna_id, structure = train_structures[global_structure_idx % len(train_structures)]
                global_structure_idx += 1
                
                try:
                    env = RNA_design_env(target_structure=structure)
                    env.rna_id = rna_id
                    env.max_steps = config['training']['max_ep_len']
                    
                    info = collect_rollout(env, ppo, buffer, config['training']['max_ep_len'])
                    
                    ep_rets.append(info['ep_ret'])
                    ep_lens.append(info['ep_len'])
                    distances.append(info['distance'])
                    
                    if info['solved']:
                        solved_count += 1
                        solved_set.add(info['rna_id'])
                    
                    steps += info['ep_len']
                    pbar.update(info['ep_len'])
                    pbar.set_postfix({
                        'ret': f"{np.mean(ep_rets[-10:]):.1f}",
                        'dist': f"{np.mean(distances[-10:]):.1f}",
                        'solved': solved_count
                    })
                    
                except Exception as e:
                    logger.warning(f"Error with structure {rna_id}: {e}")
                    continue

        epoch_end_idx = global_structure_idx -1
        
        if len(buffer) > 0:
            update_info = ppo.update(buffer)
            logger.info(f"PPO update done, epoch: {epoch}")
        else:
            logger.info("probable problem with buffer as buffer len is 0")
            update_info = {'kl': 0, 'critic_loss': 0}
        
        solve_rate = solved_count / len(ep_rets) if ep_rets else 0
        avg_ret = np.mean(ep_rets) if ep_rets else 0
        avg_len = np.mean(ep_lens) if ep_lens else 0
        avg_dist = np.mean(distances) if distances else 0
        epoch_time = time.time() - epoch_start

        start_struct_len = len(train_structures[epoch_start_idx % len(train_structures)][1])
        end_struct_len = len(train_structures[epoch_end_idx % len(train_structures)][1])
        
        
        logger.info(
            f"\nEpoch: {epoch+1:3d} | return: {avg_ret:7.1f} | length: {avg_len:7.1f}\n"
            f"distance: {avg_dist:5.1f} | solved: {solved_count}/{len(ep_rets)} ({solve_rate*100:.1f}%)\n"
            f"total solved: {len(solved_set)} | KL: {update_info.get('kl', 0):.4f} | Time: {epoch_time:.1f}s\n"
            f"Structures: idx {epoch_start_idx % len(train_structures)} (len {start_struct_len}) "
            f"to idx {epoch_end_idx % len(train_structures)} (len {end_struct_len})\n"
        )
        
        test_results = {}
        if config['evaluation'].get('evaluate_every', 10) > 0 and (epoch + 1) % config['evaluation']['evaluate_every'] == 0:
            test_results = evaluate(ppo, test_structures, config['training']['max_ep_len'], config['evaluation']['num_eval_samples'])
            logger.info(
                f"\nTesting | returns: {test_results['avg_return']:7.1f}\n"
                f"Distance: {test_results['avg_distance']:5.1f} | "
                f"Solved: {test_results['solved_count']}/{len(test_structures)} ({test_results['solve_rate']*100:.1f}%)\n"
            )
        
        if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
            ppo.save(str(log_dir / f'model_epoch_{epoch+1}.pt'))
        
        if solve_rate > best_solve_rate:
            best_solve_rate = solve_rate
            ppo.save(str(log_dir / 'best_model.pt'))
            logger.info(f"new best solving ratee: {best_solve_rate*100:.1f}%")
        
        log_file = log_dir / 'training_log.csv'
        header = not log_file.exists()
        with open(log_file, 'a') as f: # saving the files
            if header:
                f.write('epoch,avg_return,avg_length,avg_distance,solve_rate,total_solved,kl,critic_loss,test_return,test_distance,test_solve_rate\n')
            f.write(f"{epoch+1},{avg_ret:.4f},{avg_len:.2f},{avg_dist:.2f},{solve_rate:.4f},{len(solved_set)},"
                   f"{update_info.get('kl', 0):.6f},{update_info.get('critic_loss', 0):.6f},"
                   f"{test_results.get('avg_return', 0):.4f},{test_results.get('avg_distance', 0):.2f},{test_results.get('solve_rate', 0):.4f}\n")
    
    logger.info("\n" + "=" * 60)
    logger.info("final eval on the test dataset")
    logger.info("=" * 60)
    testing_structures = load_structures_from_rna_files(
        config['data']['test_dir'], config['data'].get('max_structures', None)
    )
    final_test_results = evaluate(ppo, testing_structures, config['training']['max_ep_len'])
    logger.info(
        f"final Test | return: {final_test_results['avg_return']:7.1f} | "
        f"distance: {final_test_results['avg_distance']:5.1f}\n"
        f"Solved: {final_test_results['solved_count']}/{len(test_structures)} ({final_test_results['solve_rate']*100:.1f}%)"
    )
    
    ppo.save(str(log_dir / 'final_model.pt'))
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"training complete in {total_time/60:.2f} minutes")
    logger.info(f"best train solve rate: {best_solve_rate*100:.1f}%")
    logger.info(f"final test solve rate: {final_test_results['solve_rate']*100:.1f}%")
    logger.info(f"total train solved rna structs: {len(solved_set)}/{len(train_structures)}")
    logger.info(f"models saved to: {log_dir}")
    logger.info("=" * 60)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_rl.yaml', help='path to config file')
    args = parser.parse_args()
    
    train(args.config)