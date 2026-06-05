# Transformers from Scratch

A hands-on implementation of the core components of the Transformer architecture using PyTorch — built from the ground up to understand how attention mechanisms, positional encoding, and encoder-decoder blocks work under the hood.

<p align="center">
  <img src="images/transformer-arch.png" alt="Transformer Architecture" width="300" height="400"/>
</p>

## Overview

This project provides a clean, modular, and educational implementation of the "Attention Is All You Need" architecture. By implementing the components from scratch, you can gain a deeper intuition for how self-attention, multi-head attention, masked attention, and feed-forward networks come together to form the backbone of modern Large Language Models (LLMs).

## Project Structure

The project separates the Transformer into logical, modular components:

- **`config.py`**: Central configuration (e.g., model depth, Sequence length, heads, batch size).
- **`main.py`**: The entry point that ties everything together, running a full forward and backward pass (training loop) of the model.
- **`components/`**: Core building blocks of the architecture.
  - `dataset.py`: Loads the WMT 2014 English-German dataset using pandas, builds vocabulary and creates tensor inputs.
  - `download_dataset.py`: Script leveraging `kagglehub` to download the WMT 2014 dataset.
  - `embeddings.py`: Input/Output token embeddings.
  - `positional_embeddings.py`: Sinusoidal positional encodings to provide order context.
  - `multi_head_attn.py`: Regular Multi-Head Attention (used in encoder, and cross-attention in decoder).
  - `masked_multi_head_attention.py`: Masked self-attention (used in the decoder).
  - `feed_forward.py`: Point-wise feed-forward networks (FFN).
  - `layer_norm.py`: Layer normalization modules.
  - `tokenizer.py`: A simple tokenizer for generating mock data.
  - `output_generation.py`: The final Linear/Softmax classification layer.
- **`encoder/`**:
  - `encoder.py`: The complete Encoder block which processes the input sequence.
- **`decoder/`**:
  - `decoder.py`: The complete Decoder block which processes the target sequence and attends to the encoder's output.

## Requirements

As detailed in `pyproject.toml`, the required dependencies are:
- **Python** &ge; 3.14
- **PyTorch** &ge; 2.12.0
- **NumPy** &ge; 2.4.6
- **Kagglehub & Pandas** (for downloading and processing the dataset)

*(The model will seamlessly use CUDA/GPU if available, defaulting to CPU otherwise)*

## Usage

This project uses `uv` for dependency management. First, sync the dependencies by running:

```bash
uv sync
```

You can tweak hyperparameters (like `D_MODEL`, `SEQ_LEN`, `BATCH_SIZE`, `N_HEADS`) directly in `config.py`.

To run the model and witness a full forward and backward pass (training loop):

```bash
uv run main.py
```

### What happens when you run `main.py`?
1. The script loads the WMT 2014 English-German dataset using pandas and leverages a custom tokenizer to build a vocabulary and generate input/output embeddings.
2. It generates and adds sinusoidal positional embeddings to the tokens.
3. The training loop starts (default 1000 epochs):
   - Computes fresh embeddings inside the loop to maintain a live PyTorch computation graph.
   - The inputs are passed through the `EncoderBlock`.
   - Target (output) embeddings are passed through the `DecoderBlock` alongside the context from the `encoder_output`.
   - Output probabilities are generated via the generation layer.
   - Computes expected cross-entropy loss against the one-hot encoded ground truth.
   - Performs backpropagation (`loss.backward()`) and manually steps through parameters to update weights!

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
