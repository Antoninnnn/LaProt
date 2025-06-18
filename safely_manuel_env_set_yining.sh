
module load CUDA/11.8.0 Anaconda3 ## This is for HPRC TAMU
module load GCC/10.3.0 

conda create -n laprot-dev python=3.11

source activate laprot-dev

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## install fla
pip install -U git+https://github.com/fla-org/flash-linear-attention

pip install causal-conv1d==1.5.0.post8 

## installing flame for training
git clone https://github.com/fla-org/flame.git
cd flame
pip install -e .
