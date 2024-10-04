# BART Arabic From Scratch

Implements the [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) paper from scratch using PyTorch, focused on abstractive summarization task in Arabic.

## Goal

The objective is not to create something novel but to gain a deeper understanding of transformer architectures. By applying the concepts in the paper, I aim to grasp both theoretical and practical aspects in depth.

## Data

I used the [BBC Arabic](https://github.com/csebuetnlp/xl-sum?tab=readme-ov-file) dataset for training and evaluation. It contains text-summary pairs, with 32,473 records for training and 4,689 for validation. The dataset size is too small relative to the model.

## Model
- Paper: [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461).
- Type: Transformer.
- Architecture: Encoder-Decoder.
- Size: 174M parameters.
- Language: Arabic.
- Framework: PyTorch.

## Results

The model's performance is subpar, mainly due to insufficient data. However, with larger, more suitable datasets, I am confident the model would improve significantly.

| Epoch | Loss(train) | Loss(validation) | Epoch Time (hours) | Training Time (hours) |  Device  |
|:-----:|:-----------:|:----------------:|:------------------:|:---------------------:|:--------:|
|   1   |    10.03    |       9.72       |        0.23        |          1.1          | 1 x L4OS |
|   2   |    9.61     |       9.44       |        0.22        |          1.1          | 1 x L4OS |
|   3   |    9.36     |       9.22       |        0.22        |          1.1          | 1 x L4OS |
|   4   |    9.16     |       9.05       |        0.22        |          1.1          | 1 x L4OS |
|   5   |    9.01     |       8.92       |        0.22        |          1.1          | 1 x L4OS |

## Additions

The paper used a Byte-Pair Encoding (BPE) tokenizer, but no Arabic-only BPE tokenizer was available.
So, I built one and uploaded it to Hugging Face: [arabic-bpe-tokenizer](https://huggingface.co/IsmaelMousa/arabic-bpe-tokenizer).
The model itself is also available as [arab-bart-base-174M](https://huggingface.co/IsmaelMousa/arab-bart-base-174M), with detailed documentation.

## Todo
- Fine-tune the model with a larger dataset.
- Create an inference API and integrate it with the Hugging Face Transformers library for easier use.


## Acknowledgements

This project follows the architecture and configurations from the [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) paper by [Meta AI](https://ai.meta.com/),
and I am grateful to [Lightning.AI](https://lightning.ai/) for providing free hardware resources for training.