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

# srun \
#   --ntasks=$((SLURM_NNODES * SLURM_GPUS_ON_NODE)) \
#   --ntasks-per-node=$SLURM_GPUS_ON_NODE \
#   --cpus-per-task=16 \
#   --gpus-per-task=1 \
#   --gpu-bind=closest \
#   --export=ALL,CUDA_VISIBLE_DEVICES \
#   torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=$SLURM_GPUS_ON_NODE \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     laprot/train/train_trainer_rwkv6.py