# Dataset Management Decision

## Context

This project implements a text classification model using the IMDB Large Movie Review Dataset.

Two possible approaches were considered for dataset handling:

1. Manually downloading and loading the dataset from local text files.
2. Using the HuggingFace `datasets` library to automatically download and manage the IMDB dataset.

## Decision

The project uses the manually downloaded IMDB dataset stored locally in:

data/aclImdb/

The dataset is not versioned in the Git repository. Instead, it must be downloaded separately.

## Rationale

The manual loading approach was chosen for the following reasons:

- Educational purposes: implementing the dataset pipeline from scratch improves understanding of:
  - File handling
  - Tokenization
  - Vocabulary construction
  - Encoding and padding
  - PyTorch Dataset and DataLoader internals

- Greater control over:
  - Number of samples used
  - Dataset balancing
  - Vocabulary construction strategy

- Didactic clarity, as the goal of the project is to understand Transformer architecture and NLP preprocessing fundamentals.

## Reproducibility

To reproduce the experiments:

1. Download the IMDB dataset from:
   http://ai.stanford.edu/~amaas/data/sentiment/

2. Extract it inside:
   data/aclImdb/

3. Run the training script normally.

## TLDR

Although the HuggingFace `datasets` library provides a more scalable and production-ready solution, the manual approach was intentionally maintained for educational transparency and architectural understanding.