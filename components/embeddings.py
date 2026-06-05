import pandas as pd
import torch
import torch.nn.functional as F

from config import DEVICE, D_MODEL, BATCH_SIZE, SEQ_LEN
from components.tokenizer import WordTokenizer
from components.dataset import encoder_train_input_tensors, decoder_train_input_tensors, tokenizer, token_ids, id_tokens

print(f"Total Vocabulary Size: {len(token_ids)}")
print("-" * 100)

# One-hot encodings
input_one_hot = F.one_hot(encoder_train_input_tensors, num_classes=len(token_ids)).float()
output_one_hot = F.one_hot(decoder_train_input_tensors, num_classes=len(token_ids)).float()

embedding_matrix = torch.randn(len(token_ids), D_MODEL,device=DEVICE,  requires_grad=True)
input_embeddings = torch.matmul(input_one_hot, embedding_matrix)
output_embeddings = torch.matmul(output_one_hot, embedding_matrix)