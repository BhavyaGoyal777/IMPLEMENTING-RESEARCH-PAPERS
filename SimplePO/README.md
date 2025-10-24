# SimplePO - Simple Preference Optimization

Implementation of SimplePO, a simplified and more efficient approach to preference-based optimization for aligning language models with human preferences.

## Overview

SimplePO is a streamlined alternative to DPO (Direct Preference Optimization) and other RLHF techniques, designed to align language models with human preferences through direct optimization on preference pairs while maintaining simplicity and computational efficiency.

## Files

- `SPO.ipynb` - Complete implementation of SimplePO technique (15 KB)

## Key Concepts

### Preference Optimization

The goal is to train models to generate outputs that humans prefer, using paired preference data:
- Given input prompt x
- Two responses: y_w (preferred/winner) and y_l (dispreferred/loser)
- Optimize model to increase probability of y_w relative to y_l

### SimplePO Approach

SimplePO simplifies the optimization process by:
- Direct loss formulation without complex RL frameworks
- Efficient training without reward model overhead
- Stable optimization without PPO or other RL algorithms
- Straightforward implementation and hyperparameter tuning

## Advantages

### Simplicity
- Easier to implement than full RLHF pipeline
- Fewer hyperparameters to tune
- No separate reward model training required
- Direct optimization on preference data

### Efficiency
- Reduced computational requirements
- Faster training compared to PPO-based RLHF
- Lower memory footprint
- Single-stage training process

### Stability
- More stable training dynamics
- No reward model hacking
- Reduced risk of mode collapse
- Predictable convergence behavior

### Performance
- Competitive with or better than DPO
- Effective alignment with human preferences
- Maintains model capabilities
- Improved helpfulness and safety

## Comparison with Other Methods

### vs RLHF (PPO-based)
- SimplePO: Simpler, faster, more stable
- RLHF: More complex, requires reward model, higher compute

### vs DPO
- SimplePO: Further simplified formulation
- DPO: More theoretical grounding, similar performance

### vs Supervised Fine-Tuning
- SimplePO: Better captures preference nuances
- SFT: Simpler but less aligned with preferences

## Training Process

### Data Requirements
- Pairs of responses (preferred vs dispreferred)
- Input prompts or contexts
- Preference labels (explicit or implicit)
- Can use synthetic or human-annotated data

### Loss Function
- Compares likelihoods of preferred vs dispreferred responses
- Margin-based or probability-based formulations
- Balances model performance and alignment
- Optional KL penalty to preserve base model knowledge

### Optimization
- Standard gradient descent
- Works with any autoregressive LLM
- Can be combined with other techniques (LoRA, etc.)
- Typically fewer epochs than SFT

## Applications

### Model Alignment
- Aligning with human values and preferences
- Improving helpfulness and instruction-following
- Reducing harmful or biased outputs
- Enhancing safety and reliability

### Specific Use Cases
- Chatbots and conversational AI
- Content generation with quality control
- Task-specific preference learning
- Style and tone adjustment
- Factuality improvement

## Implementation Details

### Preference Data Format
Typical structure:
- Input: prompt or conversation context
- Chosen: preferred response
- Rejected: dispreferred response
- Optional: metadata (source, quality scores)

### Training Configuration
- Learning rate: typically lower than SFT (1e-6 to 1e-5)
- Batch size: depends on model size and memory
- Epochs: usually 1-3 passes over preference data
- Beta parameter: controls strength of preference optimization

### Model Preservation
- KL divergence penalty to stay close to base model
- Prevents catastrophic forgetting
- Maintains general capabilities
- Balances alignment and performance

## Best Practices

### Data Quality
- High-quality preference pairs are crucial
- Diverse coverage of input space
- Consistent preference judgments
- Balance between different types of preferences

### Hyperparameter Tuning
- Beta: controls alignment strength
- Learning rate: affects convergence and stability
- Batch size: impacts gradient estimates
- Number of epochs: prevents overfitting

### Evaluation
- Win rate against reference model
- Human evaluation studies
- Automated preference scoring
- Task-specific benchmarks
- Safety and bias assessments

## Relation to DPO

SimplePO can be viewed as:
- A simplified variant of DPO
- Alternative loss formulation
- Similar theoretical motivation
- Often more practical for implementation

Both aim to achieve preference alignment without the complexity of traditional RLHF.
