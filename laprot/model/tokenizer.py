# tokenizer.py

import os
import json
import torch
# class ProteinTokenizer:
#     def __init__(self, max_len=512):
#         AAs = list("ACDEFGHIKLMNPQRSTVWY")  # 20 std AAs
#         extras = ["X", "B", "U", "Z", "O"]  # ambiguous, seleno, pyrro
#         specials = ["<pad>", "<bos>", "<eos>", "<mask>"]

#         # Build vocab: reserve 0 for <pad>, then other tokens
#         self.vocab = {}
#         idx = 0
#         for tok in specials:
#             self.vocab[tok] = idx
#             idx += 1
#         for aa in AAs + extras:
#             self.vocab[aa] = idx
#             idx += 1

#         self.pad_token = self.vocab["<pad>"]
#         self.bos_token = self.vocab["<bos>"]
#         self.eos_token = self.vocab["<eos>"]
#         self.mask_token = self.vocab["<mask>"]
#         self.unk_token = self.vocab["X"]  # map unknowns to 'X'
#         self.max_len = max_len

#     def encode(self, seq):
#         # clean sequence to uppercase, map unknown AAs to 'X'
#         seq = seq.upper()
#         unk = lambda aa: aa if aa in self.vocab else "X"
#         tokens = [self.bos_token] + [self.vocab.get(unk(aa), self.unk_token) for aa in seq] + [self.eos_token]

#         if len(tokens) > self.max_len:
#             tokens = tokens[:self.max_len]
#         # pad to fixed length
#         return tokens + [self.pad_token] * (self.max_len - len(tokens))

#     def tokenize_fn(self, example):
#         return {"input_ids": self.encode(example["text"])}

class ProteinTokenizer:
    def __init__(self, max_len=512):
        # ESM-compatible token list
        token_list = [
            "<cls>", "<pad>", "<eos>", "<unk>",  # special
            "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",  # 20 AAs
            "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"  # extras
        ]
        self.vocab = {tok: idx for idx, tok in enumerate(token_list)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

        # Token IDs
        self.cls_token = self.vocab["<cls>"]
        self.pad_token = self.vocab["<pad>"]
        self.eos_token = self.vocab["<eos>"]
        self.unk_token = self.vocab["<unk>"]
        self.mask_token = self.vocab["<mask>"]

        self.max_len = max_len

    def encode(self, seq, add_special_tokens=True, max_length=None, truncation=True, padding=False):
        seq = seq.upper()
        tokens = []

        if add_special_tokens:
            tokens.append(self.cls_token)

        for aa in seq:
            tokens.append(self.vocab.get(aa, self.unk_token))

        if add_special_tokens:
            tokens.append(self.eos_token)

        if max_length is None:
            max_length = self.max_len

        # Truncation
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]

        attention_mask = [1] * len(tokens)

        # Padding
        if padding and len(tokens) < max_length:
            pad_len = max_length - len(tokens)
            tokens += [self.pad_token] * pad_len
            attention_mask += [0] * pad_len

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }

    def __call__(self, seqs, padding=True, truncation=True, max_length=None, return_tensors=None):
        if isinstance(seqs, str):
            seqs = [seqs]

        # Determine dynamic max length if not provided
        if padding and max_length is None:
            max_length = max(len(seq) + 2 for seq in seqs)  # +2 for <cls> and <eos>

        # Encode each sequence with consistent max_length
        batch = [self.encode(seq, padding=padding, truncation=truncation, max_length=max_length) for seq in seqs]

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    def tokenize_fn(self, example):
        return self.encode(example["text"], padding=True)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump({
                "vocab": self.vocab,
                "max_len": self.max_len
            }, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        with open(os.path.join(load_directory, "tokenizer_config.json")) as f:
            config = json.load(f)
        tokenizer = cls(max_len=config["max_len"])
        tokenizer.vocab = config["vocab"]
        tokenizer.inv_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        return tokenizer

###our vocab

# vocab = {
#     "<cls>": 0,
#     "<pad>": 1,
#     "<eos>": 2,
#     "<unk>": 3,
#     "L": 4,
#     "A": 5,
#     "G": 6,
#     "V": 7,
#     "S": 8,
#     "E": 9,
#     "R": 10,
#     "T": 11,
#     "I": 12,
#     "D": 13,
#     "P": 14,
#     "K": 15,
#     "Q": 16,
#     "N": 17,
#     "F": 18,
#     "Y": 19,
#     "M": 20,
#     "H": 21,
#     "W": 22,
#     "C": 23,
#     "X": 24,
#     "B": 25,
#     "U": 26,
#     "Z": 27,
#     "O": 28,
#     ".": 29,
#     "-": 30,
#     "<null_1>": 31,
#     "<mask>": 32
# }

###esm vocab


# ['<cls>',
#  '<pad>',
#  '<eos>',
#  '<unk>',
#  'L',
#  'A',
#  'G',
#  'V',
#  'S',
#  'E',
#  'R',
#  'T',
#  'I',
#  'D',
#  'P',
#  'K',
#  'Q',
#  'N',
#  'F',
#  'Y',
#  'M',
#  'H',
#  'W',
#  'C',
#  'X',
#  'B',
#  'U',
#  'Z',
#  'O',
#  '.',
#  '-',
#  '<null_1>',
#  '<mask>']

###