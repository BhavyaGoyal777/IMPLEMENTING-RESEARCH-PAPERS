# Attention is All You Need

Implementation of the Transformer architecture from the paper "Attention is All You Need" by Vaswani et al. (2017).

## Overview

This implementation provides a complete Transformer model for sequence-to-sequence tasks, specifically configured for English-to-Hindi translation.

## Files

- `model1.py` - Complete Transformer architecture implementation including:
  - Input embeddings and positional encoding
  - Multi-head attention mechanism
  - Encoder and decoder blocks with residual connections
  - Feed-forward networks
  - Layer normalization
  - Projection layer for output generation

- `config1.py` - Configuration settings for model training:
  - Batch size: 8
  - Epochs: 20
  - Sequence length: 350
  - Model dimension (d_model): 512
  - Language pair: English (en) to Hindi (hi)
  - Learning rate: 1e-4

- `dataset1.py` - Dataset loading and preprocessing utilities

- `train1.py` - Training script for the Transformer model

- `requirements.txt` - Python dependencies

## Key Features

- Full encoder-decoder architecture with 6 layers
- 8 attention heads per layer
- Scaled dot-product attention
- Position-wise feed-forward networks
- Residual connections and layer normalization
- Xavier uniform weight initialization

## Model Architecture

The implementation follows the original paper's architecture:
- Encoder: 6 identical layers with multi-head self-attention and feed-forward networks
- Decoder: 6 identical layers with masked self-attention, encoder-decoder attention, and feed-forward networks
- Model dimension: 512
- Feed-forward dimension: 2048
- Dropout: 0.1

## Usage

Configure the model parameters in `config1.py`, then run the training script to train the model on your translation dataset.
