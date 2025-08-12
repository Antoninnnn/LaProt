import torch
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments, default_data_collator
from fla.models import RWKV6ForCausalLM, RWKV6Config
from transformers import AutoModelForCausalLM
from laprot.model.tokenizer import ProteinTokenizer
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm

# Load tokenized dataset
ds = load_from_disk("data/uniref_1m_tokenized/")
# Convert 'input_ids' and create labels
ds = ds.map(lambda ex: {"labels": ex["input_ids"][1:], "input_ids": ex["input_ids"][:-1]}, 
            remove_columns=["input_ids"], batched=False)

# Model setup
tok = ProteinTokenizer()

# Define tokenizer stub for Trainer API
class DummyTokenizer:
    pad_token_id = tok.pad_token


config = RWKV6Config(
    vocab_size=len(tok.vocab),
    hidden_size=512,
    num_hidden_layers=12,
    num_heads=8,
    intermediate_size=512 * 3,
    hidden_act="sqrelu",
    norm_first=True,
    fuse_norm=True,
    norm_bias=False,
    norm_eps=1e-5,
    attn_mode="chunk",
    expand_k=0.5,      # numeric, not None
    expand_v=1.0,      # numeric, not None
    proj_low_rank_dim=32,
    gate_low_rank_dim=64,
        # ✅ Special token ids
    pad_token_id=tok.pad_token,
    bos_token_id=tok.cls_token,
    eos_token_id=tok.eos_token,
)


# from fla.modules.layernorm_gated import RMSNormGated

# def patch_norm(module, hidden_size):
#     for name, child in module.named_children():
#         if isinstance(child, RMSNormGated):
#             setattr(module, name, RMSNormGated(hidden_size, eps=child.eps, norm_before_gate=False))
#         else:
#             patch_norm(child, hidden_size)

# model = Mamba2ForCausalLM(config).cuda()
model = AutoModelForCausalLM.from_config(config).cuda()
# patch_norm(model, hidden_size=config.hidden_size)


# def patch_block_norms(model, config):
#     for i, block in enumerate(model.backbone.layers):
#         correct_dim = config.hidden_size * config.expand
#         if block.norm.weight.shape[0] != correct_dim:
#             block.norm = RMSNorm(correct_dim, eps=config.norm_eps)
#             print(f"[Patched] block {i} RMSNorm to dim {correct_dim}")

# patch_block_norms(model, config)


# # Test
# B, L, D = 2, 64, 1024
# x = torch.randn(B, L, D)
# gate = torch.randn(B, L, D)

# norm = RMSNormGated(D)
# out = norm(x, gate)

loss_fn = FusedCrossEntropyLoss(ignore_index=tok.pad_token, reduction="mean")


#######################
# input_ids = torch.randint(0, model.config.vocab_size, (2, 64)).cuda()
# print("input_ids.shape:", input_ids.shape)
# inputs_embeds = model.backbone.embeddings(input_ids)
# print("inputs_embeds.shape:", inputs_embeds.shape)  # expect [2,64,512]

# block = model.backbone.layers[0]
# x = block.norm(inputs_embeds)
# print("After norm:", x.shape)  # Expected: [2, 64, 512]

# x_mixed = block.mixer(x)
# print("After mixer:", x_mixed.shape)  # Expected: [2, 64, 512]
################

class ProteinTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").cuda()
        input_ids = inputs.pop("input_ids").cuda()
        outputs = model(input_ids=input_ids)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels_flat = labels.view(-1)
        loss = self.loss_fn(logits, labels_flat)
        return (loss, outputs) if return_outputs else loss

        
training_args = TrainingArguments(
    output_dir="laprot_rwkv6_protein_sm_data_lr_4e-4",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    bf16=True,
    learning_rate=4e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    warmup_ratio=0.03,
    logging_steps=100,
    save_steps=9000,
)

trainer = ProteinTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=default_data_collator,
    tokenizer=tok,
)

trainer.loss_fn = loss_fn  # inject the fused loss function

# print("norm.weight.shape:", model.norm.weight.shape)


if __name__ == "__main__":
    trainer.train()