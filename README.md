# nanomech-rna

A reinforcement learning framework for RNA sequence design using Graph Neural Networks (GNNs) with a two stage training approach: Behavioral Cloning (BC) for imitation learning from expert solutions, followed by Proximal Policy Optimization (PPO) for reinforcement learning.  The model learns to predict nucleotide mutations at specific locations to design RNA sequences that fold into target secondary structures.  Built with PyTorch Geometric for graph based representations of RNA structures.

## Data

Download the required data files: 
- **The BC Dataset is already in the repo, link to original dataset**: [Download from the Repo](https://github.com/eternagame/EternaBrain) - Extract to `./X5/`
- **RL Training Data**: [Download from the Repo](https://github.com/automl/learna) - you can find the intructions to download the Rfam Learn dataset in the readme, Extract to `./data_rl`

If you put your datasets in folders with different names, please update the cofig yaml files accordingly.

## Installation

### 1. Create a new conda environment

```bash
conda create -n nanomech-rna python=3.12
conda activate nanomech-rna
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Inference (on pretrained model)

Put your RNA sequence in dot bracket notation in inputs.txt file

Then run:
```bash
python main.py
```
Your answer would be in outputs.txt file

If you want to change the input and output files, please change in the config_inference.yaml file
Please make sure that the network architecture in the config files of Behavioral Cloning, Reinforcement Learning and Inference is same.

## Training

### Stage 1: Behavioral Cloning (BC)

```bash
python bc_main.py
```

### Stage 2: Reinforcement Learning (PPO)

Make sure that the network architecture is same as the Behavioral Cloning, by looking at the config files.
Add the path to the saved BC pretrained model in the config_rl.yaml file.

```bash
python train.py
```

> **Note**: RL training automatically loads the BC pretrained weights from `./checkpoints/bc/best_model. pt`