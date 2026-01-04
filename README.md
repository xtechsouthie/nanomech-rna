# nanomech-rna

A reinforcement learning framework for RNA sequence design using Graph Neural Networks (GNNs) with a two stage training approach: **Behavioral Cloning (BC)** for imitation learning from expert solutions, followed by **Proximal Policy Optimization (PPO)** for reinforcement learning. The model learns to predict nucleotide mutations at specific locations to design RNA sequences that fold into target secondary structures. Built with PyTorch Geometric for graph based representations of RNA structures.

## Data

### BC Training Data (EternaBrain)
- **Already included** in this repo under `./X5/`
- Original dataset: [EternaBrain on GitHub](https://github.com/eternagame/EternaBrain)
- Contains expert human solutions to RNA design puzzles

### RL Training Data (Rfam Learn)
- Download from: [LEARNA on GitHub](https://github.com/automl/learna)
- Follow instructions in their README to download the Rfam Learn dataset
- Extract to `./data_rl/rfam_learn/`

## Installation

### 1. Create a new conda environment

```bash
conda create -n nanomech-rna python=3.12
conda activate nanomech-rna
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Inference (On Pretrained Model)

1. **Prepare your input**: Add RNA structures (dot bracket notation) to `inputs.txt`, one per line:
```
(((((......))))) 
(((((((....(((...........)))((((((((..(((((((((((((((((((...(((((......))))).)))))).)))))))))))))..))))))))..)))))))
```

2. **Run inference**:
 ```bash
python main.py
```

3. **Check results**: Designed sequences will be saved to `outputs.txt`

## Training

### Stage 1: Behavioral Cloning (BC)

Train the model to imitate expert solutions :
Update the config_bc.yaml to select the network configurations.

```bash
python bc_main.py
```

### Stage 2: Reinforcement Learning (PPO)

Fine tune the model using RL. 
change the epochs, the length of puzzles to train on, data directory, etc in the config_rl.yaml file.

**Important:** Before running, ensure:
1. `config_rl.yaml` network architecture matches `config_bc.yaml`
2. `training.pretrained` points to your BC checkpoint (e.g., `./checkpoints/bc/best_model.pt`)

```bash
python train.py
```

> **Note**: RL training automatically loads the BC pretrained weights from `./checkpoints/bc/best_model. pt`