import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Architecture (Paper Base Spec) ---
D_MODEL = 512        # Dimension of model
N_HEADS = 8          # Number of Attention Heads
D_HEAD = D_MODEL // N_HEADS # 64
ENCODER_LAYERS = 1    # Paper specification
DECODER_LAYERS = 1    # Paper specification

# --- Data Handling ---
BATCH_SIZE = 128      # Sentence batching capacity for A100
SEQ_LEN = MAX_LEN = 128 # Captures full context without aggressive cutting

# --- Optimizer & LR Scheduler Spec ---
# Total steps calculation based on dataset size: 
# (100,000 sentences / 128 batch_size) * 30 epochs = ~23,400 steps
TOTAL_STEPS = 1000   
WARMUP_STEPS = 0.04 * TOTAL_STEPS   # Strict paper value for stability

BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-9