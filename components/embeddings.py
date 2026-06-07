import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE, D_MODEL, BATCH_SIZE, SEQ_LEN
from components.tokenizer import WordTokenizer
from components.dataset import encoder_train_input_tensors, decoder_train_input_tensors, tokenizer, token_ids, id_tokens
from components.positional_embeddings import get_positional_embeddings

print(f"Total Vocabulary Size: {len(token_ids)}")
print("-" * 100)

# print(f' Encoder input tensors sample:\n {encoder_train_input_tensors}')
# print(f' Decoder input tensors sample:\n {decoder_train_input_tensors}')

class Embeddings(nn.Module):
    def __init__(self, vocab_size=len(token_ids), d_model=D_MODEL):
        super().__init__()
        self.embedding_matrix = torch.nn.Embedding(vocab_size, d_model, device=DEVICE)
        self.position_embeddings = get_positional_embeddings(SEQ_LEN, DEVICE, d_model)
        self.dropout = torch.nn.Dropout(0.1).to(DEVICE)  # Adding dropout for regularization

    def forward(self, x):
        return self.dropout(self.embedding_matrix(x) + self.position_embeddings)