# Maximizing Surrogate Objective Function: DeepSeekMath

## Objective

This repository provides an implementation to maximize the surrogate objective function as mentioned in the DeepSeekMath paper. The approach leverages Symbolic Programming and natural language processing (NLP) models to simulate the optimization of the target function efficiently.

## Prerequisites

Before running the code, ensure that the following dependencies are installed:

1. **PyTorch**: The framework used for deep learning computations.
2. **BERT Base Cased**: A pre-trained transformer model required for tokenization.

## Installation Steps

### Step 1: Install PyTorch

Follow the official PyTorch installation guide based on your system configuration: [PyTorch Installation](https://pytorch.org/get-started/locally/)

```bash
pip install torch torchvision torchaudio
```

### Step 2: Setup BERT Base Cased Tokenizer

To use the BERT tokenizer, install the `transformers` library and load the model:

```bash
pip install transformers
```

Download and load the BERT Base Cased model:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

## Usage

Run the main script to execute the optimization process:

```bash
python GRPO_Sim.py
```

## References

If you use this repository in your work, please cite the corresponding paper:

> **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** - Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo, 2024. [[Link to Paper]](https://arxiv.org/abs/2402.03300)

## License

This project is licensed under the [MIT License](LICENSE).

