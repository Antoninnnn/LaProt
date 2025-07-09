
# module purge
# module load CUDA/11.8.0 Anaconda3
# # eval "$(conda shell.bash hook)"

# # For multi nodes training, you need to set the following packages :
# ml GCC/10.3.0 

# export PATH=$SCRATCH/local/pdsh/bin:$PATH

# # Set CUDA environment variables on Grace HPRC of TAMU
# export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# # Set Hugging Face cache directory to a writable location
# export HF_HOME=$SCRATCH/hf_cache

# #Set the Torch cache directory in the $SCRATCH

# export TORCH_HOME=$SCRATCH/.cache/torch
# export TRITON_CACHE_DIR=/scratch/user/yining_yang/triton_cache
# export TORCHINDUCTOR_CACHE_DIR=/scratch/user/yining_yang/torchinductor_cache


# export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
# export MASTER_PORT=29500
# export NCCL_DEBUG=INFO
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# # export WANDB_API_KEY=c6da89ba565a8b25f5b18c6fb722e7ad6637d4de  # from wandb.ai/settings
# # export WANDB_MODE=offline  # or remove this if online logging is available
# # export WANDB_DIR=$SCRATCH/wandb_logs

# # # echo "[INFO] Writing hostfile:"
# # # scontrol show hostnames $SLURM_NODELIST | sed 's/$/ slots=2/' > scripts/hostfile.txt
# # echo "[INFO] MASTER_ADDR: $MASTER_ADDR"
# # echo "[INFO] MASTER_PORT: $MASTER_PORT"
# # echo "[INFO] Nodes:"
# # scontrol show hostnames $SLURM_NODELIST

# # module load cuda python
# source activate laprot-dev



from transformers import AutoTokenizer, AutoModelForCausalLM
from fla.models import RWKV6ForCausalLM, RWKV6Config
from laprot.model.tokenizer import ProteinTokenizer

import torch

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load Model and Tokenizer ===
ckpt_dir = "/scratch/user/yining_yang/TAMU/PhD/LaProt/laprot_rwkv6_protein/checkpoint-15625"
tokenizer = ProteinTokenizer.from_pretrained(ckpt_dir)
model = RWKV6ForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).cuda()
model.eval()

# === Input Sequences ===
sequences = [
    "MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGP",
    "MKKLLFAIPLVVPFYSHSPLGDK",
    "GSDVIVTVEATAYFKEALGK"
]

# === Tokenize ===
batch = tokenizer(sequences, return_tensors="pt", padding=True)
input_ids = batch["input_ids"].cuda()
attention_mask = batch["attention_mask"].cuda()

# === Run Inference ===
with torch.no_grad():
    outputs = model(input_ids=input_ids, output_hidden_states=1)
# Now this is safe
if outputs.hidden_states is not None:
    hidden = outputs.hidden_states[-1]  # final layer
else:
    hidden = outputs.logits  # fallback
# === Extract Mean Embeddings (ignoring padding) ===
mask = attention_mask.unsqueeze(-1)  # shape: [B, T, 1]
embeddings = (hidden * mask).sum(1) / mask.sum(1)

# === t-SNE Visualization ===
tsne = TSNE(n_components=2, perplexity=2, random_state=42)

# # Convert embeddings to float32 and numpy for t-SNE
# emb_2d = tsne.fit_transform(embeddings.cpu().numpy())
emb_np = embeddings.to(torch.float32).cpu().numpy()
emb_2d = tsne.fit_transform(emb_np)

# === Plot ===
plt.figure(figsize=(6, 5))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=["red", "blue", "green"])

for i, seq in enumerate(sequences):
    plt.text(emb_2d[i, 0], emb_2d[i, 1], f"Seq {i+1}", fontsize=9)


plt.title("t-SNE of RWKV6 Protein Embeddings")
plt.grid(True)
plt.tight_layout()
plt.savefig("asset/imgs/rwkv_protein_embedding")