# ⚡ Linear Attention Protein Model (LaProt)

> A linear attention framework for modeling protein sequences and multimodal token states (e.g., 3Di structures) as language-like inputs.

---

## 🧠 Motivation

Protein sequences and their multimodal token sequences (like 3Di structural tokens) can be interpreted as **different modalities or languages**.

Given the **limited vocabulary** of proteins (e.g., just 20 amino acids), **state tracking** becomes a key property in protein language modeling. Proteins dynamically transition between functional states:

- 🔁 **Folding Pathway**  
  Proteins pass through intermediate states as they fold:  
  `unfolded → partially folded → native structure`

- 🔗 **Allostery**  
  Binding at one site (e.g., ligand or another protein) alters the protein's state at a distant site.

- 🧬 **Post-translational Modifications (PTMs)**  
  Events like **phosphorylation** or **methylation** modify residues and change the protein's function.

> These dynamic transitions are often **encoded by specific motifs or residue positions** in the sequence.

---

## 🚀 Release Timeline

| Date     | Update Description                            |
|----------|-----------------------------------------------|
| 06/2025  | 🟢 Environment Setup & Toy experiments |

---

## ⚙️ Installation Guide

### 1. 🐍 Create Conda Environment

```bash
conda create -n laprot-dev python=3.11
conda activate laprot-dev
```

### 2. 🔧 Install PyTorch with CUDA 11.8
```bash  
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. ⚡ Install Flash Linear Attention (FLA)

```bash
pip install -U git+https://github.com/fla-org/flash-linear-attention
pip install causal-conv1d==1.5.0.post8 
```

### 4. 🔥 Install FLAME for Training

```bash
git clone https://github.com/fla-org/flame.git
cd flame
pip install -e .
```

📌 Notes

- This project assumes compatibility with CUDA 11.8 and PyTorch 2.7.0.

- FLA and FLAME are optimized for high-throughput linear attention and efficient protein modeling.


(
  
  Persional note:

```
# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch

# # Create the directory if it doesn't exist
# mkdir -p $TRANSFORMERS_CACHE


```




)

## 📁 Repository Structure (To be added)

```bash
LaProt/
├── configs/
├── data/
├── 
├── scripts/
├── laprot/
│   ├── __init__.py
│   └── model
└── README.md

```
## Training

Use the code from `train/train_trainer_rwkv6.py` and the following script from `scripts/run_pretrain.sh`:

```shell
#!/bin/bash
#SBATCH --job-name=laprot_pretrain_dev
#SBATCH --output=logs/laprot_pretrain_dev_%j.out
#SBATCH --error=logs/laprot_pretrain_dev_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=24:00:00




module purge
module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"

# For multi nodes training, you need to set the following packages :
ml GCC/10.3.0 

export PATH=$SCRATCH/local/pdsh/bin:$PATH

# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=/scratch/user/yining_yang/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/scratch/user/yining_yang/torchinductor_cache


export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export WANDB_API_KEY=c6da89ba565a8b25f5b18c6fb722e7ad6637d4de  # from wandb.ai/settings
export WANDB_MODE=offline  # or remove this if online logging is available
export WANDB_DIR=$SCRATCH/wandb_logs

# echo "[INFO] Writing hostfile:"
# scontrol show hostnames $SLURM_NODELIST | sed 's/$/ slots=2/' > scripts/hostfile.txt
echo "[INFO] MASTER_ADDR: $MASTER_ADDR"
echo "[INFO] MASTER_PORT: $MASTER_PORT"
echo "[INFO] Nodes:"
scontrol show hostnames $SLURM_NODELIST

# module load cuda python
source activate laprot-dev

RANK=$(echo $(hostname) | awk -v master="$MASTER_ADDR" '{if ($0 == master) print 0; else print 1}')


torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=2 \
  --node_rank=$RANK \
  --rdzv_backend=static \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  laprot/train/train_trainer_rwkv6.py


```

## Inference
Here we show a demo how to run the rwkv6 model in python:

```python

import torch
from fla.models import RWKV6ForCausalLM
from your_tokenizer_module import ProteinTokenizer  # Replace with your actual path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load Model and Tokenizer ===
ckpt_dir = "/scratch/user/yining_yang/TAMU/PhD/LaProt/laprot_rwkv6_protein/checkpoint-15625"
tokenizer = ProteinTokenizer.from_pretrained(ckpt_dir)
model = RWKV6ForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).cuda()
model.eval()

sequences = [
    "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGP",
    "MKKLLFAIPLVVPFYSHSPLGDK",
    "GSDVIVTVEATAYFKEALGK"
]

batch = tokenizer(sequences, return_tensors="pt", padding=True)
print(batch["input_ids"].shape)  # e.g. [3, 34]


with torch.no_grad():
    outputs = model(input_ids=input_ids, output_hidden_states=True)

# Make sure to set output_hidden_states=True for the forward to enable the hidden states output
if outputs.hidden_states is not None:
    hidden = outputs.hidden_states[-1]  # final layer
else:
    hidden = outputs.logits  # fallback
```
We could print the hidden size:
```shell
>>> model.eval()
RWKV6ForCausalLM(
  (model): RWKV6Model(
    (embeddings): Embedding(33, 512, padding_idx=1)
    (layers): ModuleList(
      (0): RWKV6Block(
        (pre_norm): LayerNorm(512, eps=1e-05)
        (attn_norm): LayerNorm(512, eps=1e-05)
        (attn): RWKV6Attention(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (x_proj): Sequential(
            (0): LerpLinear(512, 160)
            (1): Tanh()
            (2): Linear(in_features=160, out_features=512, bias=False)
          )
          (r_proj): DDLerpLinear(512, 256)
          (w_proj): DDLerpLinear(512, 256, low_rank_dim=64)
          (k_proj): DDLerpLinear(512, 256)
          (v_proj): DDLerpLinear(512, 512)
          (g_proj): DDLerpLinear(512, 512)
          (g_norm): GroupNorm(8, 512, eps=1e-05)
          (o_proj): Linear(in_features=512, out_features=512, bias=False)
        )
        (ffn_norm): LayerNorm(512, eps=1e-05)
        (ffn): RWKV6FeedForward(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): LerpLinear(512, 1536)
          (value): Linear(in_features=1536, out_features=512, bias=False)
          (receptance): LerpLinear(512, 512)
        )
      )
      (1-11): 11 x RWKV6Block(
        (attn_norm): LayerNorm(512, eps=1e-05)
        (attn): RWKV6Attention(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (x_proj): Sequential(
            (0): LerpLinear(512, 160)
            (1): Tanh()
            (2): Linear(in_features=160, out_features=512, bias=False)
          )
          (r_proj): DDLerpLinear(512, 256)
          (w_proj): DDLerpLinear(512, 256, low_rank_dim=64)
          (k_proj): DDLerpLinear(512, 256)
          (v_proj): DDLerpLinear(512, 512)
          (g_proj): DDLerpLinear(512, 512)
          (g_norm): GroupNorm(8, 512, eps=1e-05)
          (o_proj): Linear(in_features=512, out_features=512, bias=False)
        )
        (ffn_norm): LayerNorm(512, eps=1e-05)
        (ffn): RWKV6FeedForward(
          (time_shift): ZeroPad2d((0, 0, 1, -1))
          (key): LerpLinear(512, 1536)
          (value): Linear(in_features=1536, out_features=512, bias=False)
          (receptance): LerpLinear(512, 512)
        )
      )
    )
    (norm): LayerNorm(512, eps=1e-05)
  )
  (lm_head): Linear(in_features=512, out_features=33, bias=False)
)

## Use the hidden as the embedding 
>>> hidden.shape
torch.Size([3, 33, 512])
```
The hidden dimensionality is 512