# Karpathy Neural Networks

Educational implementations of neural networks following Andrej Karpathy's teaching style and approach to building language models from scratch.

## Overview

This directory contains implementations inspired by Andrej Karpathy's neural network tutorials, focusing on building character-level and word-level language models with clear, educational code.

## Files

- `bigramModel.ipynb` - Implementation of a simple bigram language model (536 KB)
- `NNappliedtoBigramModel.ipynb` - Neural network enhancement of the bigram model (550 KB)
- `names.txt` - Training dataset containing names (228 KB)
- `test.txt` - Test dataset

## Approach

These implementations follow a pedagogical approach:
- Build models from first principles
- Clear, readable PyTorch code
- Step-by-step progression from simple to complex
- Focus on understanding core concepts

## Models

### Bigram Model
- Simplest form of language modeling
- Predicts next character based on current character
- Baseline for understanding language model fundamentals
- Uses probability distributions learned from training data

### Neural Network Bigram Model
- Enhances basic bigram with neural network layers
- Learns distributed representations of characters
- Demonstrates embedding concepts
- Shows how neural networks improve over statistical methods

## Learning Objectives

- Understanding language model fundamentals
- Character-level text generation
- Probability distributions and sampling
- Neural network basics in NLP context
- Training loops and optimization
- Embedding layers and their role

## Dataset

The names.txt dataset contains a collection of names used for:
- Training character-level language models
- Learning character distributions and patterns
- Generating new, similar names
- Understanding sequence modeling basics

## Use Cases

- Learning language model fundamentals
- Character-level text generation
- Understanding neural network basics
- Educational exploration of NLP concepts
