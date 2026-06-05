import torch
from config import SEQ_LEN


class WordTokenizer:
    def __init__(self):
        # self.max_len = max_len
        # Reserve the first few IDs for special control tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token
            ]

        # Initialize vocabulary mappings
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, sentences):
          for sentence in list(sentences):
            for word in self.special_tokens + str(sentence).split():
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
                    self.id2word[len(self.id2word)] = word

    def encode(self, sentence, add_sos=False, add_eos=False, max_len=None):
        """Converts a raw text string into a fixed-length list of token IDs."""
        words = str(sentence).split()
        tokens = []

        if add_sos:
            tokens.append(self.word2id[self.sos_token])

        for word in words:
            # Fallback to PAD if a word isn't in vocab (or handle as <UNK>)
            tokens.append(self.word2id.get(word, self.word2id[self.unk_token]))

        if add_eos:
            tokens.append(self.word2id[self.eos_token])

        # Truncate if the sequence is longer than max_len
        if max_len is not None:
          tokens = tokens[:max_len]

        # Pad with 0s if the sequence is shorter than max_len
          if len(tokens) < max_len:
            padding_len = max_len - len(tokens)
            tokens.extend([self.word2id[self.pad_token]] * padding_len)

        return tokens

    def decode(self, token_ids):
        words = []
        for idx in token_ids:
            word = self.id2word.get(idx, self.unk_token)
            if word == self.pad_token:
                continue
            words.append(word)
        return " ".join(words)