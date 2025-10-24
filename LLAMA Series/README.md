# LLaMA Series

Implementations of Meta's LLaMA (Large Language Model Meta AI) models, including LLaMA 2 and LLaMA 3 architectures.

## Overview

LLaMA is a family of large language models developed by Meta AI, designed to be efficient, performant, and open for research use. This directory contains implementations of multiple versions of the LLaMA architecture.

## Files

- `llama2.ipynb` - Implementation of LLaMA 2 architecture (20 KB)
- `llama3.ipynb` - Implementation of LLaMA 3 architecture (21 KB)

## Architecture Highlights

### Core Components
- Decoder-only Transformer architecture
- Pre-normalization using RMSNorm
- SwiGLU activation function in feed-forward layers
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA) in newer versions

### LLaMA 2 Features
- Trained on 2 trillion tokens
- Context length: 4096 tokens
- Available in 7B, 13B, 70B parameter variants
- Improved training stability
- Enhanced instruction-following capabilities

### LLaMA 3 Features
- Extended context length (8192+ tokens)
- Improved tokenizer with larger vocabulary
- Better multilingual support
- Enhanced reasoning capabilities
- Optimized inference performance

## Key Improvements Across Versions

### LLaMA 2 Improvements
- Better safety and alignment
- Enhanced instruction tuning
- Reduced hallucinations
- Improved factual accuracy

### LLaMA 3 Enhancements
- Superior performance on reasoning tasks
- Better code generation capabilities
- Improved long-context understanding
- Enhanced multilingual performance

## Technical Details

### Attention Mechanism
- Multi-head attention with efficient implementations
- Query, key, value projections
- Attention masking for causal language modeling

### Normalization
- RMSNorm instead of LayerNorm for efficiency
- Applied before attention and feed-forward blocks

### Position Encoding
- RoPE (Rotary Position Embeddings)
- Relative position information
- Better extrapolation to longer sequences

## Applications

- Text generation and completion
- Question answering
- Summarization
- Code generation
- Instruction following
- Chat and dialogue systems
- Fine-tuning for specialized tasks

## Implementation Notes

The notebooks provide:
- Complete model architecture implementation
- Attention mechanism details
- Forward pass logic
- Inference examples
- Model loading and configuration
