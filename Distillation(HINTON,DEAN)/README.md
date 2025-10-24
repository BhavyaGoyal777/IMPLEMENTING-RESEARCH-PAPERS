# Knowledge Distillation (Hinton & Dean)

Implementation of the knowledge distillation technique introduced by Geoffrey Hinton and Jeff Dean in "Distilling the Knowledge in a Neural Network" (2015).

## Overview

Knowledge distillation is a model compression technique where a smaller "student" network learns to mimic the behavior of a larger, more complex "teacher" network.

## Files

- `distillation.ipynb` - Jupyter notebook with complete implementation of knowledge distillation

## Subdirectories

- `FDA/` - Additional implementations related to feature distillation and advanced techniques

## Key Concepts

Knowledge distillation works by:
- Training a large, high-capacity teacher model on the task
- Using the teacher's softened probability distributions (soft targets) as supervision
- Training a smaller student model to match both hard labels and soft targets
- Using temperature scaling to soften the probability distributions
- Transferring "dark knowledge" from teacher to student

## Advantages

- Model compression: Smaller models with similar performance
- Faster inference: Reduced computational requirements
- Deployment efficiency: Suitable for resource-constrained environments
- Knowledge transfer: Student learns from teacher's generalization patterns

## Temperature Parameter

The temperature parameter T controls the softness of probability distributions:
- Higher T: Softer distributions, more information about similarities between classes
- T = 1: Standard softmax
- Training uses high T for distillation, inference uses T = 1

## Applications

- Mobile and edge deployment
- Real-time inference systems
- Model ensemble compression
- Transfer learning scenarios
