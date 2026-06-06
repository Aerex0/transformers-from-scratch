import pandas as pd
import torch
from components.tokenizer import WordTokenizer
from components.download_dataset import path
from config import DEVICE, MAX_LEN

# Load the dataset into pandas DataFrames
train_df = pd.read_csv(
    path + "/wmt14_translate_de-en_train.csv",
    engine='python',
    nrows=10,
    on_bad_lines='skip'
)

val_df = pd.read_csv(
    path + "/wmt14_translate_de-en_validation.csv",
    engine='python',
    on_bad_lines='skip'
)

test_df = pd.read_csv(
    path + "/wmt14_translate_de-en_test.csv",
    engine='python',
    on_bad_lines='skip'
)

tokenizer = WordTokenizer()
tokenizer.build_vocab(train_df['en'] + ' ' + train_df['de'])

token_ids = tokenizer.word2id
id_tokens = tokenizer.id2word

train_en_list = train_df['en'].values
train_de_list = train_df['de'].values

encoder_train_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in train_en_list]
encoder_train_input_tensors = torch.tensor(encoder_train_input_tokens).to(DEVICE)

decoder_train_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in train_de_list]
decoder_train_input_tensors = torch.tensor(decoder_train_input_tokens).to(DEVICE)


# Validation
val_en_list = val_df['en'].values
val_de_list = val_df['de'].values

enocder_val_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in val_en_list]
encoder_val_input_tensors = torch.tensor(enocder_val_input_tokens).to(DEVICE)

decoder_val_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in val_de_list]
decoder_val_input_tensors = torch.tensor(decoder_val_input_tokens).to(DEVICE)

# Test

test_en_list = test_df['en'].values
test_de_list = test_df['de'].values

encoder_test_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in test_en_list]
encoder_test_input_tensors = torch.tensor(encoder_test_input_tokens).to(DEVICE)

decoder_test_input_tokens = [tokenizer.encode(sentence, add_sos=True, add_eos=True, max_len=MAX_LEN) for sentence in test_de_list]
decoder_test_input_tensors = torch.tensor(decoder_test_input_tokens).to(DEVICE)