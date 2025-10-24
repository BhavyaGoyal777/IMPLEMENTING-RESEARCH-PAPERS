# ViT - Vision Transformer

Implementation of Vision Transformer (ViT) from the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2020).

## Overview

Vision Transformer (ViT) applies the Transformer architecture, originally designed for NLP, directly to images by treating image patches as tokens. This revolutionary approach demonstrated that pure Transformer models can achieve excellent performance on image classification tasks without convolutional layers.

## Files

- `ViT.ipynb` - Complete Vision Transformer implementation notebook (160 KB)
- `ViT-cifar100.ipynb` - ViT model trained on CIFAR-100 dataset (163 KB)
- `ViT.py` - Modular Python implementation with reusable components (5 KB)

## Architecture

### Core Components

#### Patch Embedding
- Splits image into fixed-size patches (typically 16x16)
- Flattens patches into 1D vectors
- Linearly projects patches to embedding dimension
- Implementation uses Conv2d for efficient patch extraction

#### Position Embeddings
- Learnable 1D position embeddings added to patch embeddings
- Provides positional information to the model
- Enables the model to understand spatial relationships
- Can be interpolated for different image resolutions

#### Transformer Encoder
- Stack of transformer encoder blocks
- Multi-head self-attention mechanism
- Layer normalization (applied before attention and MLP)
- Residual connections around each sub-layer
- MLP (feed-forward) blocks with GELU activation

#### Classification Head
- Special [CLS] token prepended to sequence
- [CLS] token representation used for classification
- MLP head for final prediction
- Often includes LayerNorm and dropout

## Key Features

### Patch-Based Processing
- Divides image into non-overlapping patches
- Each patch treated as a token
- Standard 16x16 patch size for ViT-Base
- Smaller patches (14x14, 8x8) for higher resolution

### Self-Attention for Vision
- Global receptive field from first layer
- Learns spatial relationships across entire image
- No inductive biases of convolutions
- Attention maps reveal what model focuses on

### Scalability
- Scales well with model size and data
- Pre-training on large datasets crucial for performance
- Transfer learning to downstream tasks
- Efficient parallel processing

## Model Variants

### ViT-Base (ViT-B/16)
- 12 transformer layers
- Hidden dimension: 768
- MLP dimension: 3072
- 12 attention heads
- Patch size: 16x16
- Parameters: ~86M

### ViT-Large (ViT-L/16)
- 24 transformer layers
- Hidden dimension: 1024
- MLP dimension: 4096
- 16 attention heads
- Parameters: ~307M

### ViT-Huge (ViT-H/14)
- 32 transformer layers
- Hidden dimension: 1280
- MLP dimension: 5120
- 16 attention heads
- Patch size: 14x14
- Parameters: ~632M

## Training Strategy

### Pre-training
- Large-scale datasets (ImageNet-21k, JFT-300M)
- High resolution images
- Data augmentation crucial
- Long training schedules (thousands of epochs)

### Fine-tuning
- Transfer to downstream tasks
- Higher resolution than pre-training
- Position embedding interpolation
- Task-specific classification heads

### Data Augmentation
- RandAugment
- Mixup and CutMix
- Random cropping and flipping
- Color jittering

## Implementation Details

### PatchEmbeddings Class
Located in ViT.py:5, implements efficient patch extraction using Conv2d:
- Kernel size = stride = patch size
- Single convolution operation splits and projects patches
- More efficient than manual patch extraction and linear projection

### Positional Encoding
- 1D learnable embeddings
- Added to patch embeddings after projection
- Shape: (1, num_patches + 1, hidden_dim)
- "+1" accounts for [CLS] token

### Attention Mechanism
- Scaled dot-product attention
- Multi-head for different representation subspaces
- Dropout for regularization
- Attention weights can be visualized

## CIFAR-100 Training

The ViT-cifar100.ipynb notebook includes:
- Model adaptation for 32x32 images
- Training on 100-class classification
- Hyperparameter tuning for smaller images
- Performance evaluation and analysis
- Comparison with CNN baselines

## Advantages

### Global Context
- Self-attention provides global receptive field from layer 1
- CNNs require many layers for global context
- Better long-range dependency modeling

### Flexibility
- Easy to adapt to different input sizes
- Position embeddings can be interpolated
- Unified architecture for various vision tasks

### Scalability
- Performance improves with model size
- Benefits from large-scale pre-training
- Efficient on modern hardware (GPUs/TPUs)

## Disadvantages

### Data Hungry
- Requires large pre-training datasets
- Underperforms CNNs on small datasets (without pre-training)
- Strong inductive biases of CNNs help with limited data

### Computational Cost
- Quadratic complexity with number of patches
- Higher memory requirements than CNNs
- Longer training times

## Applications

### Image Classification
- ImageNet and other classification benchmarks
- Transfer learning to custom datasets
- Fine-grained recognition tasks

### Other Vision Tasks
- Object detection (DETR-style)
- Semantic segmentation
- Image generation (as part of diffusion models)
- Self-supervised learning

### Multimodal
- Vision-language models (CLIP, ALIGN)
- Visual question answering
- Image captioning

## Key Insights from Paper

1. **Pre-training Scale Matters**: ViT benefits significantly from large-scale pre-training
2. **Fewer Inductive Biases**: Pure Transformer can learn from data what CNNs encode architecturally
3. **Patch Size Trade-offs**: Smaller patches increase accuracy but also computational cost
4. **Position Embeddings**: Even simple 1D embeddings work well for 2D images

## Comparison with CNNs

### ViT Strengths
- Global context from first layer
- Better scaling properties
- Unified architecture across modalities
- Strong transfer learning

### CNN Strengths
- Better with limited data
- Translation equivariance built-in
- Local feature extraction
- Computational efficiency

## Best Practices

### For Small Datasets
- Use pre-trained models
- Heavy data augmentation
- Smaller model variants
- Consider hybrid CNN-Transformer models

### For Large-Scale Training
- Large batch sizes
- Layer-wise learning rate decay
- Warmup and cosine learning rate schedule
- Mixed precision training

### For Inference
- Batch processing when possible
- Optimize attention implementation
- Consider quantization
- Cache position embeddings

## Related Work

- DeiT (Data-efficient Image Transformers): Distillation techniques for ViT
- Swin Transformer: Hierarchical vision transformer with shifted windows
- CvT (Convolutional vision Transformer): Combines convolutions with transformers
- BEiT: Masked image modeling for self-supervised pre-training
