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

### **Model Architecture**  
The core building block of a large language model is the **Transformer**, which consists of:  

- **Attention Mechanisms** for contextual token interactions, with the following variants:  
  - **Single-head self-attention**  
  - **Multi-head self-attention** (used in modern transformers)  
  - **Causal (masked) self-attention** to ensure each token attends only to previous tokens  

- **Feed-forward Networks (FFN)** with two common implementations:  
  - **Multi-layer perceptron (MLP)** – A standard dense feed-forward network  
  - **Mixture of Experts (MoE)** – Uses multiple expert networks and a gating mechanism for efficiency and scalability  

- **Layer Normalization** for stabilizing training.  

### **Training Components**  
- Tokenization and positional encoding  
- Loss functions (typically cross-entropy for next-token prediction)  
- Optimization algorithms (e.g., AdamW)  
- Learning rate scheduling and regularization  

### **Evaluation & Generation**  
- Perplexity measurement for model evaluation  
- Sampling strategies for text generation (e.g., greedy decoding, nucleus sampling)  

This repository provides a clean, educational codebase for understanding how GPT models function under the hood.  
