# train_demo.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from fla.layers import mamba2
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from laprot.model.tokenizer import ProteinTokenizer
from transformers import AutoModelForCausalLM

import deepspeed

# Load tokenized dataset saved with save_to_disk()
ds = load_from_disk("data/uniref_1m_tokenized/")
ds = ds.shuffle(seed=42)  # optional

# def collate_fn(batch):
#     xs = torch.tensor([b["input_ids"][:-1] for b in batch], dtype=torch.long)
#     ys = torch.tensor([b["input_ids"][1:] for b in batch], dtype=torch.long)
#     return xs, ys



def collate_fn(batch):
    input_seqs = [torch.tensor(b["input_ids"][:-1], dtype=torch.long) for b in batch]
    target_seqs = [torch.tensor(b["input_ids"][1:], dtype=torch.long) for b in batch]
    
    xs = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    ys = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    
    return xs, ys


dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Model setup
tok = ProteinTokenizer()

from fla.models import Mamba2Config

config = Mamba2Config(
    vocab_size=len(tok.vocab),  # e.g. ~24 (20 AAs + X + special tokens)
    hidden_size=512,            # lowers memory footprint compared to default 2048
    num_hidden_layers=12,       # medium depth for prototyping
    num_heads=8,                # head_dim = hidden_size / num_heads = 64
    head_dim=64,
    state_size=64,              # smaller state to balance complexity
    expand=2,                   # MLP expansion
    chunk_size=256,             # good default for chunkwise SSM
    conv_kernel=4,
    fuse_cross_entropy=True,
    fuse_norm=True,
    rms_norm=True,
    use_bias=False,
    use_conv_bias=True,
    rescale_prenorm_residual=True,
    residual_in_fp32=True,
    time_step_rank=64,
    time_step_min=0.001,
    time_step_max=0.1,
)

model =  AutoModelForCausalLM.from_config(config)

loss_fn = FusedCrossEntropyLoss(ignore_index=0, reduction='mean')

# DeepSpeed initialization
model_engine, optimizer, _, _ = deepspeed.initialize(
    config="laprot/train/ds_config.json",
    model=model,
    model_parameters=model.parameters(),
)

def fix_rmsnorm_shapes(model):
    for name, param in model.named_parameters():
        if "rmsnorm_weight" in name and param.dim() > 1:
            print(f"Fixing {name} from shape {param.shape} to {(param.numel(),)}")
            with torch.no_grad():
                param.data = param.view(-1)
            # Optional: also register the new shape for consistency
            param._is_view = False  # avoid in-place modification bugs

fix_rmsnorm_shapes(model)

# Training loop
for epoch in range(3):
    total_loss = 0.0
    for x, y in dl:
        x, y = x.cuda(), y.cuda()
        try:
            logits = model_engine(x)
        except Exception as e:
            print(f"Input shape: {x.shape}")
            raise e
        # loss = fce(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        model_engine.backward(loss)
        model_engine.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss = {total_loss/len(dl):.4f}")