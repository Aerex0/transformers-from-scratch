import pandas as pd
import torch
import torch.nn.functional as F

from config import DEVICE, D_MODEL, BATCH_SIZE, SEQ_LEN
from components.tokenizer import WordTokenizer
from components.dataset import encoder_train_input_tensors, decoder_train_input_tensors, tokenizer, token_ids, id_tokens

print(f"Total Vocabulary Size: {len(token_ids)}")
print("-" * 100)

# One-hot encodings
# input_one_hot = F.one_hot(encoder_train_input_tensors, num_classes=len(token_ids)).float()
# output_one_hot = F.one_hot(decoder_train_input_tensors, num_classes=len(token_ids)).float()

# embedding_matrix = torch.randn(len(token_ids), D_MODEL,device=DEVICE,  requires_grad=True)
embedding_matrix = torch.nn.Embedding(len(token_ids), D_MODEL, device=DEVICE) # We will keep use these same weights for input, output and final linear layer in output generation.

print(f' Encoder input tensors shape: {encoder_train_input_tensors.shape}, Decoder input tensors shape: {decoder_train_input_tensors.shape}')
print(f' Embedding matrix shape: {embedding_matrix.weight.shape}')
# print(f' Encoder input tensors sample:\n {encoder_train_input_tensors}')
# print(f' Decoder input tensors sample:\n {decoder_train_input_tensors}')


print(f' Embedding Matrix of input tokens:\n {embedding_matrix(encoder_train_input_tensors)}')