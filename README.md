# Music Generation Transformers

This project implements a MIDI sequence model using a causal transformer architecture.

## Model

The core model is a decoder-style transformer for symbolic music generation.

- MIDI events are tokenized with the REMI scheme.
- A learned token embedding layer maps tokens to vectors.
- A learned positional embedding layer encodes sequence position.
- Multiple transformer blocks are stacked, each containing:
  - multi-head causal self-attention using `scaled_dot_product_attention`
  - linear projection and dropout after attention
  - a feedforward network with ReLU activation and dropout
  - layer normalization and residual connections
- The final linear layer projects back to the vocabulary for next-token prediction.

## Training method

Training is performed as autoregressive next-token prediction on tokenized MIDI data.

- The dataset loader constructs random context windows of fixed `block_size`.
- Each batch contains token sequences and shifted targets for the next-token objective.
- The loss is cross-entropy between model logits and the next token in the sequence.
- Optimization uses AdamW with a cosine-like learning rate schedule and gradient clipping.
- Mixed precision is enabled on CUDA with `bfloat16`.

## Fine-tuning with DPO

A separate flow uses preference-based learning with Direct Preference Optimization (DPO).

- A generator model creates candidate continuations from the same prompt.
- A judge model scores each generated sequence.
- Pairs of winner/loser continuations are selected based on score difference.
- A frozen reference model is used for comparison.
- The active model is updated with DPO loss computed from active and reference log-probabilities for winner and loser sequences.

This architecture supports both standard autoregressive music generation and preference-driven fine-tuning through DPO.