from config import DEVICE, D_MODEL, SEQ_LEN
import torch
import math

debug = False

def get_positional_embeddings(SEQ_LEN, DEVICE, D_MODEL):

    position = torch.arange(SEQ_LEN, dtype=torch.float, device=DEVICE).unsqueeze(1) # (seq_len, 1)
    div_term = torch.arange(0, D_MODEL, 2, dtype=torch.float, device=DEVICE)
    div_term = torch.exp(div_term * (-(math.log(10000.0))) / D_MODEL) # (1, d_model/2)
    pos_emb = torch.zeros(SEQ_LEN, D_MODEL, device=DEVICE)
    pos_emb[:, 0::2] = torch.sin(position * div_term) # For Even indices
    pos_emb[:, 1::2] = torch.cos(position * div_term) # For Odd indices
    pos_emb = pos_emb.unsqueeze(0) # This will add a fake dimension (1, )
    if debug:
        print(f'Positional Embeddings:\n {pos_emb}')
        print(f'Positional Embeddings shape: {pos_emb.shape}')

    return pos_emb