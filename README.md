# LLM-from-Scratch

This repository serves as a personal learning note for understanding the fundamental 
concepts of large language models (LLMs), including some implementations built from scratch.

## Usage

```bash
source setup.sh
```
The script will create a conda environment with the name `llm-from-scratch` and install the dependencies.

## The Building Blocks of a Large Language Model  

To implement a minimal GPT model from scratch, we need the following key components:  

### 1. Model Architecture  
The core building block of a large language model is the **Transformer**, which consists of:  

- **Attention Mechanisms** for contextual token interactions, with the following variants:  
  - **Single-head self-attention**  
  - **Multi-head self-attention** (used in modern transformers)  
  - **Causal (masked) self-attention** to ensure each token attends only to previous tokens  

- **Feed-forward Networks (FFN)** with two common implementations:  
  - **Multi-layer perceptron (MLP)** – A standard dense feed-forward network  
  - **Mixture of Experts (MoE)** – Uses multiple expert networks and a gating mechanism for efficiency and scalability  

- **Layer Normalization** for stabilizing training.  

### 2. Training Components  
- Tokenization and positional encoding  
- Loss functions (typically cross-entropy for next-token prediction)  
- Optimization algorithms (e.g., AdamW)  
- Learning rate scheduling and regularization  

### 3. Evaluation & Generation  
- Perplexity measurement for model evaluation  
- Sampling strategies for text generation (e.g., greedy decoding, nucleus sampling)  

This repository provides a clean, educational codebase for understanding how GPT models function under the hood. This part is implemented in `gpt2-inference.py`.

## Training Steps

This part is based on the nice [tutorial](https://github.com/jingyaogong/minimind?tab=readme-ov-file).

### 1. Pretraining
The foundation phase where the model learns language patterns from massive text corpora (Wikipedia, books, web content) using self-supervised learning. The primary objective is next-token prediction (causal language modeling), enabling the model to understand language structure and acquire broad knowledge. The loss function is the cross-entropy loss:
$$
L = - \sum_{k=1}^{N} \log P(x_k | x_1, x_2, \cdots, x_{k-1})
$$
where $P(x_k | x_1, x_2, \cdots, x_{k-1})$ is the probability of the next token $x_k$ given the previous tokens $x_1, x_2, \cdots, x_{k-1}$.

### 2. Supervised Fine-Tuning (SFT)
This phase transforms a pretrained model into an instruction-following assistant. The model is fine-tuned on high-quality examples of instructions paired with desired responses, teaching it to understand and follow human instructions rather than simply continuing text.

### 3. Reinforcement Learning from Human Feedback (RLHF)
RLHF aligns the model with human preferences and values. It typically involves:
- Training a reward model on human preference data
- Using reinforcement learning (often PPO - Proximal Policy Optimization) to optimize the model against this reward
- Alternatively, Direct Preference Optimization (DPO) provides a more stable approach by directly optimizing from preferences without a separate reward model

### 4. Knowledge Distillation (KD)
A technique to create smaller, more efficient models by transferring knowledge from larger "teacher" models to compact "student" models. The student learns to mimic the teacher's output distributions rather than just the final predictions, preserving nuanced knowledge while reducing computational requirements.

### 5. Parameter-Efficient Fine-Tuning (PEFT)
Methods like LoRA (Low-Rank Adaptation), QLoRA, and adapter-based approaches allow for efficient adaptation of large models by training only a small subset of parameters. These techniques significantly reduce memory requirements and training time while maintaining performance, making domain-specific adaptation more accessible.

### 6. Specialized Training for Reasoning
Advanced techniques to enhance specific capabilities like reasoning, planning, and tool use. This may involve chain-of-thought prompting, specialized datasets with step-by-step reasoning, and techniques that encourage the model to break down complex problems into manageable steps.

This part is implemented in `gpt2-training.py`.
