import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 6
BATCH_SIZE = 2
SEQ_LEN = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS
LR = 1