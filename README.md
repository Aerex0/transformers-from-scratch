# Transformers from Scratch

A hands-on implementation of the core components of the Transformer architecture using PyTorch — built from the ground up to understand how attention mechanisms, positional encoding, and encoder-decoder blocks work under the hood.

<p align="center">
  <img src="images/transformer-arch.png" alt="Transformer Architecture" width="300" height="400"/>
</p>

## Overview

> **⚠️ Disclaimer:** This project currently only implements the architecture and forward pass of the Transformer model. No training loop, backward pass, or weight optimization has been implemented as of now. It outputs un-trained (random) predictions to demonstrate the data flow.

This project provides a clean, modular, and educational implementation of the "Attention Is All You Need" architecture. By implementing the components from scratch, you can gain a deeper intuition for how self-attention, multi-head attention, masked attention, and feed-forward networks come together to form the backbone of modern Large Language Models (LLMs).

## Project Structure

The project separates the Transformer into logical, modular components:

- **`config.py`**: Central configuration (e.g., model depth, Sequence length, heads, batch size).
- **`main.py`**: The entry point that ties everything together, running a full forward pass of the model.
- **`components/`**: Core building blocks of the architecture.
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

*(The model will seamlessly use CUDA/GPU if available, defaulting to CPU otherwise)*

## Usage

This project uses `uv` for dependency management. First, sync the dependencies by running:

```bash
uv sync
```

You can tweak hyperparameters (like `D_MODEL`, `SEQ_LEN`, `BATCH_SIZE`, `N_HEADS`) directly in `config.py`.

To run a forward pass of the model and see the transformation from tokenized inputs to final predicted tokens:

```bash
uv run main.py
```

### What happens when you run `main.py`?
1. The script retrieves input and output embeddings for a sample text.
2. It generates and adds 2D positional embeddings to the tokens.
3. The inputs are passed through the `EncoderBlock`.
4. Target (output) embeddings are prepared, augmented with positional signals, and passed through the `DecoderBlock` alongside the `encoder_output`.
5. Finally, output probabilities are generated via the generation layer, an `argmax` yields predicted token IDs, and those IDs are decoded back to text sentences. 

## References
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
