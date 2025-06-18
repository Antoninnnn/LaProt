# Linear Attention Protein Model (LaProt)

Protein sequences and its multimodal token sequences(3di s) could be requided as different languages. Considering the limited state of protein representation(for example only 20 amino acids in its sequence), the state tracking property becomes significant for protein language modeling. 

Proteins undergo conformational changes, folding, and binding events—each of these can be seen as a state change:

---    Folding pathway: As a protein sequence folds, it passes through intermediate states (e.g., unfolded → partially folded → native structure).

---    Allostery: Binding at one site (e.g., a ligand or another protein) can change the state of the protein elsewhere.

---    Post-translational modifications (PTMs): Like phosphorylation or methylation, which modify specific residues, effectively changing the protein’s functional state.

These changes often depend on specific motifs or residue positions in the sequence, meaning the primary sequence encodes state transition potential.


## Release 

[06/2025] Initial Project Repo Created & Environment setup


## Installation



The environment was built with cuda118 and torch 2.7.0. We rely on fla to implement the linear attention model and flame to train target model. 
```
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
```
