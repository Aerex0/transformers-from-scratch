import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 32 # Dimension of model must be divisible by number of heads (N_HEADS)
BATCH_SIZE = 4
SEQ_LEN = MAX_LEN = 25
N_HEADS = 8
D_HEAD = D_MODEL // N_HEADS
EPOCHS = 1000
ENCODER_LAYERS = 1
DECODER_LAYERS = 1
WARMUP_STEPS = 0.04 * EPOCHS
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9