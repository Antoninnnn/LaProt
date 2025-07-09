
module load CUDA/11.8.0 Anaconda3 ## This is for HPRC TAMU
module load GCC/10.3.0 

conda create -n laprot-dev python=3.11

source activate laprot-dev

conda deactivate 

source activate laprot-dev

export PYTHONPATH=/scratch/user/yining_yang/python-packages:$PYTHONPATH

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## install fla
pip install -U git+https://github.com/fla-org/flash-linear-attention

pip install --no-build-isolation causal-conv1d==1.5.0.post8 
# # or using its git repo
# git clone https://github.com/Dao-AILab/causal-conv1d.git
# cd causal-conv1d
# pip install .

pip install --no-build-isolation mamba-ssm
# # or using its git repo
# git clone https://github.com/state-spaces/mamba.git
# cd mamba
# pip install --no-build-isolation .

## installing flame for training
git clone https://github.com/fla-org/flame.git



cd flame
pip install -e .
