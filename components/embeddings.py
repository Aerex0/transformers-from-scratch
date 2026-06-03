import torch
import torch.nn.functional as F
from config import DEVICE, D_MODEL, BATCH_SIZE, SEQ_LEN
from components.tokenizer import WordTokenizer

# 1. Encoder Inputs (English Source)
encoder_sentences = [
    "the cat is sleeping",
    "the dog is running",
    "the cat is running"
]

# 2. Decoder Inputs (French Targets)
decoder_sentences = [
    "le chat dort",
    "le chien court",
    "le chat court"
]

tokenizer = WordTokenizer(max_len=SEQ_LEN)
# input_vocab = tokenizer.build_vocab(encoder_sentences)
# output_vocab = tokenizer.build_vocab(decoder_sentences)

vocab = tokenizer.build_vocab(encoder_sentences + decoder_sentences)
vocab_size = len(tokenizer.word2id)

print(f"Total Vocabulary Size: {vocab_size}")
print("Vocabulary Map:", tokenizer.word2id)
print("-" * 50)

input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True) for sentence in encoder_sentences]
input_tensors = torch.tensor(input_tokens).to(DEVICE)
output_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True) for sentence in decoder_sentences]
output_tensors = torch.tensor(output_tokens).to(DEVICE)

# print(f'----> Input Tokens:\n {input_tensors}')
# print(f'----> Output Tokens:\n {output_tensors}')

# One-hot encodings
input_one_hot = F.one_hot(input_tensors, num_classes=len(tokenizer.word2id)).float()
output_one_hot = F.one_hot(output_tensors, num_classes=len(tokenizer.word2id)).float()

# print(f'----> Input One-Hot Encodings:\n {input_one_hot}')
# print(f'----> Output One-Hot Encodings:\n {output_one_hot}')


embedding_matrix = torch.randn(len(tokenizer.word2id), D_MODEL,device=DEVICE,  requires_grad=True)
input_embeddings = torch.matmul(input_one_hot, embedding_matrix)
output_embeddings = torch.matmul(output_one_hot, embedding_matrix)
# print(f'----> Input Embeddings:\n {input_embeddings}')
# print(f'----> Output Embeddings:\n {output_embeddings}')

# # These are Dummy embeddings
# input_embeddings = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)
# output_embeddings = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL).to(DEVICE)

# if __name__ == "__main__":
#     print(input_embeddings)
#     print(output_embeddings)
