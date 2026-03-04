# Dataset Processing Strategy

## Context

The original IMDB Large Movie Review Dataset contains 50,000 reviews (25,000 for training and 25,000 for testing). 

Training on the full dataset may increase computational cost and training time, especially when running experiments locally without GPU acceleration.

## Sampling Strategy

To make local experimentation feasible while preserving statistical validity, a balanced subset of the dataset is used.

The sampling strategy ensures:

- Equal number of positive and negative samples
- Preservation of class distribution
- Reduced training time
- Lower memory consumption

Additionally, sequence length is limited through truncation to control computational complexity, since Transformer models have quadratic complexity with respect to sequence length.

## Dataset Preparation

The `prepare_dataset` function is responsible for:

- Reducing the dataset size
- Applying preprocessing steps
- Performing tokenization
- Converting tokens to numerical representations
- Applying padding or truncation as required

The design goal is to process data dynamically rather than permanently storing large processed files.

## Storage and Data Flow

Currently, processed data is temporarily saved in: `processed_train.pt`

However, this file acts only as an intermediate artifact and is not intended to be version-controlled.

In practice:

- The dataset class stores the necessary components in memory
- The `__getitem__()` method dynamically preprocesses and encodes samples
- The `DataLoader` returns tensors ready for model consumption

This design keeps the pipeline modular and allows experimentation with preprocessing strategies without permanently modifying the raw dataset.

## Design Considerations

Although saving fully processed datasets can improve loading speed, the current approach prioritizes:

- Transparency of preprocessing steps
- Flexibility for experimentation
- Clear separation between raw data and processed tensors
- Educational understanding of the full NLP pipeline
