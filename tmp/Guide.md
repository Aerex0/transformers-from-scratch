# 🤖 Transformer From Scratch — Learning Roadmap

> Build bottom-up. Check each box only when you've understood *and* implemented the concept, not just read about it.

---

## Phase 0 — Prerequisites

- [ ] Comfortable with matrix multiplication (shapes, batched matmul)
- [ ] Understand softmax — what it does, why it saturates
- [ ] Know backpropagation intuitively (chain rule, gradients flowing backward)
- [ ] PyTorch basics — tensors, `nn.Module`, `forward()`, `autograd`
- [ ] Read **"Attention Is All You Need"** (Vaswani et al., 2017) — at least once, even loosely

---

## Phase 1 — The Core Primitive: Scaled Dot-Product Attention

- [ ] Understand what Q, K, V represent conceptually (query, key, value as a soft lookup)
- [ ] Understand *why* we scale by `√d_k` (prevent softmax saturation in high dimensions)
- [ ] Write the attention formula by hand: `softmax(QKᵀ / √d_k) · V`
- [ ] Implement `scaled_dot_product_attention(Q, K, V)` — inputs/outputs are just tensors
- [ ] Verify output shape is correct for a dummy input
- [ ] Add **masking** support (for decoder's causal/autoregressive mask)
- [ ] Visualize the attention weights matrix for a toy input — does it make intuitive sense?

---

## Phase 2 — Multi-Head Attention

- [ ] Understand *why* multiple heads exist (different subspaces, different relationships)
- [ ] Understand the projection matrices: `W_Q`, `W_K`, `W_V`, `W_O`
- [ ] Implement `MultiHeadAttention` as an `nn.Module`
  - [ ] Linear projections for Q, K, V
  - [ ] Split into `h` heads, reshape tensors correctly
  - [ ] Run attention per head (reuse Phase 1 function)
  - [ ] Concatenate heads, apply output projection `W_O`
- [ ] Verify: input shape `(batch, seq_len, d_model)` → output shape same
- [ ] Check that `d_model = h × d_k` holds in your implementation

---

## Phase 3 — Position-wise Feed-Forward Network

- [ ] Understand what this layer does (applied independently per position)
- [ ] Know the standard dimensions: `d_model → d_ff → d_model` (typically `d_ff = 4 × d_model`)
- [ ] Implement `FeedForward(d_model, d_ff)` — two linear layers + ReLU (or GELU)
- [ ] Verify shapes pass through correctly

---

## Phase 4 — Positional Encoding

- [ ] Understand *why* transformers need positional info (attention is permutation-invariant)
- [ ] Understand the sinusoidal formula — different frequencies per dimension
- [ ] Understand the intuition: each position gets a unique "fingerprint" across dimensions
- [ ] Implement `PositionalEncoding(d_model, max_len)` — compute the encoding matrix
- [ ] Plot the encoding matrix as a heatmap — you should see a wave pattern
- [ ] Add it to token embeddings in a forward pass

---

## Phase 5 — Encoder Block

- [ ] Understand the sub-layer structure: attention → add & norm → FFN → add & norm
- [ ] Understand **residual connections** — why they help (gradient flow, representation preservation)
- [ ] Understand **Layer Normalization** — normalize across features, not batch
- [ ] Implement `EncoderBlock(d_model, num_heads, d_ff, dropout)`
  - [ ] Self-attention sublayer with residual + LayerNorm
  - [ ] FFN sublayer with residual + LayerNorm
- [ ] Stack N encoder blocks into a full `Encoder`

---

## Phase 6 — Decoder Block

- [ ] Understand the three sublayers in decoder (masked self-attn, cross-attn, FFN)
- [ ] Understand **masked self-attention** — why causal masking prevents peeking at future tokens
- [ ] Understand **cross-attention** — Q comes from decoder, K/V come from encoder output
- [ ] Implement `DecoderBlock(d_model, num_heads, d_ff, dropout)`
  - [ ] Masked self-attention sublayer
  - [ ] Cross-attention sublayer (takes encoder output as input)
  - [ ] FFN sublayer
  - [ ] Residuals + LayerNorm on all three
- [ ] Stack N decoder blocks into a full `Decoder`

---

## Phase 7 — Full Encoder-Decoder Transformer

- [ ] Implement token embedding layer (`nn.Embedding`)
- [ ] Wire together: Embedding + PositionalEncoding → Encoder → Decoder → Linear + Softmax
- [ ] Implement `Transformer(src_vocab, tgt_vocab, d_model, N, h, d_ff, dropout)`
- [ ] Do a full forward pass with dummy data — no errors, shapes are correct
- [ ] Count total parameters — sanity check against known model sizes

---

## Phase 8 — Training on a Toy Task

- [ ] Pick a simple task: **sequence copy** (output = input) or **sequence reversal**
- [ ] Build a tiny dataset + dataloader for the task
- [ ] Implement training loop with `CrossEntropyLoss`
- [ ] Use **Adam optimizer** with the warmup learning rate schedule from the paper
- [ ] Implement **teacher forcing** during training
- [ ] Train until the model solves the toy task (loss converges, outputs are correct)
- [ ] Write a simple greedy decode / inference function

---

## Phase 9 — Reflect & Extend

- [ ] Re-read the paper — things that were confusing before should now be obvious
- [ ] Compare your implementation against a reference (Annotated Transformer, nanoGPT)
- [ ] Understand what changes to make it a **decoder-only** model (GPT-style)
- [ ] Understand what changes to make it an **encoder-only** model (BERT-style)
- [ ] (Optional) Swap sinusoidal PE for **RoPE** or **ALiBi** and understand the difference

---

## Reference Checkpoints

| Milestone | Signal that you're ready to proceed |
|---|---|
| After Phase 1 | You can explain attention to someone else without notes |
| After Phase 5 | Encoder forward pass runs on random input without shape errors |
| After Phase 6 | Full seq2seq forward pass works end-to-end |
| After Phase 8 | Model solves the copy/reversal task reliably |
| After Phase 9 | You could build GPT or BERT from here with minor modifications |

---

*Based on "Attention Is All You Need" — Vaswani et al., 2017*
