#!/bin/bash
#SBATCH --job-name=pannot_pretrain
#SBATCH --output=logs/laprot_jupyternotebook_%j.out
#SBATCH --error=logs/laprot_jupyternotebook_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=6:00:00




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


source activate laprot-dev

jupyter notebook
