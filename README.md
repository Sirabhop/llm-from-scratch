# llm-from-scratch

This repo aim to demonstrate my understanding on the large languauge model.

## Code Structure

### Import Libraries:

Imports essential libraries for working with PyTorch (a deep learning framework).

### Hyperparameters:

Defines parameters that control model training and behavior:
- block_size: Length of text chunks used for training.
- step_size: How much the window of text shifts after each prediction.
- batch_size: Number of text chunks processed simultaneously.
- max_iter: Maximum training steps.
- eval_iters: How often to evaluate the model during training.
- dropout: A technique to prevent overfitting (the model memorizing the training data too closely).

### Load Data:

- Reads the wizard_of_oz.txt file.
- Creates a vocabulary of unique characters and dictionaries for mapping characters to integers and vice versa.

### Data Preparation:

- get_batch: A function to fetch batches of text data for training and evaluation.
- estimate_loss: A function to calculate the model's prediction error during training.

### Bigram Language Model:

BigramLanguageModel: A simple language model that predicts the next character based only on the previous character (a bigram).

Includes methods for:
- forward: Making predictions and calculating loss.
- generate: Generating text.

### Training:

1. Creates a BigramLanguageModel.
2. Defines an optimizer (torch.optim.AdamW) to adjust model parameters during training.
3. get_model_output: generates text with given input
4. Trains the model for a specified number of iterations (max_iter), periodically evaluating its performance.

### Enhanced GPT Language Model (GPTLanguageModel):

- Block: A building block of the GPT architecture, containing self-attention mechanisms and feed-forward layers.
- Head: A single attention head used in the multi-head attention mechanism.
- MultiHeadAttention: Combining multiple attention heads for a richer understanding of text.
- FeedForward: Standard neural network layers for additional processing.
- GPTLanguageModel: The main GPT model, using the above components. It includes:
- Embedding layers for tokens and positions.
- Multiple blocks for deeper processing.
- Layer normalization and a linear layer for final prediction.

Methods for:
- _init_weights: Initializing model parameters.
- forward: Making predictions and calculating loss.
