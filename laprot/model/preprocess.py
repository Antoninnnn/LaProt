from datasets import load_dataset
from tokenizer import ProteinTokenizer
# from laprot.model.tokenizer import ProteinTokenizer
from torch.utils.data import DataLoader

import torch


tok = ProteinTokenizer()
ds = load_dataset("bloyal/uniref50", split="train")

tokenized_input_ids= ds.map(tok.tokenize_fn)


# # Store the input ids 
# tokenized.save_to_disk("data/preprocessed_uniref/uniref50_tokenized/")


# # load the input ids 
# from datasets import load_from_disk
# ds = load_from_disk("uniref_tokenized_arrow/")

###################


def collate_fn(batch):
    xs = torch.tensor([b["input_ids"][:-1] for b in batch], dtype=torch.long)
    ys = torch.tensor([b["input_ids"][1:] for b in batch], dtype=torch.long)
    return xs, ys

dl = DataLoader(
    tokenized.with_format("torch"),
    batch_size=32,
    collate_fn=collate_fn,
)