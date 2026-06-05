import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 352 # Dimension of model must be divisible by number of heads (N_HEADS)
BATCH_SIZE = 2
SEQ_LEN = MAX_LEN = 25
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS
LR = 0.1
EPOCHS = 1000