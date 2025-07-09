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