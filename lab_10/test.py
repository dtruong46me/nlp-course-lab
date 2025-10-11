# -*- coding: utf-8 -*-
"""
Refactored script for training a sequence-to-sequence machine translation model
from English to Vietnamese using TensorFlow and Keras, with an attention mechanism.
"""

import os
import re
import time
import unicodedata
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================

# --- Data Configuration ---
DATA_DIR = "data_iwslt15"
SITE_PREFIX = "https://nlp.stanford.edu/projects/nmt/data"
DATA_FILES = {
    "train": ("train.en", "train.vi"),
    "dev": ("tst2012.en", "tst2012.vi"),
    "test": ("tst2013.en", "tst2013.vi"),
}
NUM_EXAMPLES = 50000  # Number of training examples to use
MAX_SENTENCE_LENGTH = 50 # Max number of tokens in a sentence

# --- Model Hyperparameters ---
BUFFER_SIZE = 32000
BATCH_SIZE = 64
EMBEDDING_DIM = 512
HIDDEN_UNITS = 512
EPOCHS = 5 # Increased for slightly better results

# --- Training Configuration ---
CHECKPOINT_DIR = './model_checkpoints'


# ==============================================================================
# 2. Data Preparation
# ==============================================================================
def preprocess_sentence(sentence: str) -> str:
    """
    Add <start> and <end> tokens to the sentence.
    """
    return f"<start> {sentence.strip()} <end>"

def tokenize_sentences(
        sentences: List[str]
) -> Tuple[tf.Tensor, tf.keras.preprocessing.text.Tokenizer]: # type: ignore
    """
    Tokenize and pad a list of sentences.
    Args:
        sentences (List[str]): List of sentences to tokenize.
    Returns:
        Tuple[tf.Tensor, tf.keras.preprocessing.text.Tokenizer]: A tuple containing the padded tensor and fitted tokenizer
    """
    # Create a tokenizer and fit on the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='') # type: ignore
    
    # Fit the tokenizer on the sentences
    tokenizer.fit_on_texts(sentences)

    # Convert sentences to sequences and pad them
    tensor = tokenizer.texts_to_sequences(sentences)

    # Pad the sequences to ensure uniform length
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post') # type: ignore

    return tensor, tokenizer

def load_data(
        source_path: str,
        target_path: str,
        num_examples: int = None,
) -> Tuple[List[str], List[str]]:
    """
    Load and preprocess sentence pairs from file paths.
    Args:
        source_path (str): Path to the source language file.
        target_path (str): Path to the target language file.
        num_examples (int, optional): Number of examples to load. If None, load all.
    Returns:
        Tuple[List[str], List[str]]: Lists of preprocessed source and target sentences.
    """
    # Read source sentences
    with open(source_path, 'r', encoding='utf-8') as f:
        source_sentences = f.readlines()

    # Read target sentences
    with open(target_path, 'r', encoding='utf-8') as f:
        target_sentences = f.readlines()

    # Assuming both files have the same number of lines
    assert len(source_sentences) == len(target_sentences)

    # Get the number of examples to use
    if num_examples:
        source_sentences = source_sentences[:num_examples]
        target_sentences = target_sentences[:num_examples]

    # Preprocess sentences
    source_data, target_data = [], []
    for src, tgt in zip(source_sentences, target_sentences):
        if len(src.split()) <= MAX_SENTENCE_LENGTH and len(tgt.split()) <= MAX_SENTENCE_LENGTH:
            source_data.append(preprocess_sentence(src))
            target_data.append(preprocess_sentence(tgt))

    return source_data, target_data


# ==============================================================================
# 3. Model Architecture (Seq2Seq with Attention)
# ==============================================================================

class Encoder(tf.keras.Model): # type: ignore
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: int, batch_size: int):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # type: ignore
        self.gru = tf.keras.layers.GRU( # type: ignore
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x: tf.Tensor, hidden: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self) -> tf.Tensor:
        return tf.zeros((self.batch_size, self.hidden_units))

class BahdanauAttention(tf.keras.layers.Layer): # type: ignore
    """Bahdanau Attention layer."""
    def __init__(self, units: int):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units) # type: ignore
        self.W2 = tf.keras.layers.Dense(units) # type: ignore
        self.V = tf.keras.layers.Dense(1) # type: ignore

    def call(self, query: tf.Tensor, values: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values # type: ignore
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights # type: ignore

class Decoder(tf.keras.Model): # type: ignore
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_units: int, batch_size: int):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.hidden_units)

    def call(
        self, x: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

# ==============================================================================
# 4. Training Setup
# ==============================================================================

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the loss, masking padded values.

    Args:
        real: The true target tensor.
        pred: The predicted logits from the model.

    Returns:
        The mean loss for the batch.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(
    source: tf.Tensor, target: tf.Tensor, enc_hidden: tf.Tensor,
    encoder: Encoder, decoder: Decoder, target_tokenizer: tf.keras.preprocessing.text.Tokenizer
) -> tf.Tensor:
    """Performs a single training step."""
    loss = 0.0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(source, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing: feeding the target as the next input
        for t in range(1, target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(target[:, t], predictions)
            # Use teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = loss / int(target.shape[1])
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# ==============================================================================
# 5. Evaluation and Visualization
# ==============================================================================

def evaluate(
    source_sentence: str, encoder: Encoder, decoder: Decoder,
    source_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    target_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    max_len_source: int, max_len_target: int
) -> Tuple[str, str, np.ndarray]:
    """
    Translates a source sentence and returns the result and attention plot.
    """
    attention_plot = np.zeros((max_len_target, max_len_source))
    
    clean_sentence = preprocess_sentence(source_sentence)
    inputs = [source_tokenizer.word_index.get(i, 0) for i in clean_sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_len_source, padding='post'
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, HIDDEN_UNITS))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    for t in range(max_len_target):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = target_tokenizer.index_word.get(predicted_id, '')
        
        if predicted_word == '<end>':
            return result, clean_sentence, attention_plot

        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, clean_sentence, attention_plot

def plot_attention(attention: np.ndarray, sentence: List[str], predicted_sentence: List[str]):
    """Plots the attention weights."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    """Main function to run the data pipeline, training, and evaluation."""
    # --- 1. Load and Prepare Data ---
    download_data()
    
    train_src_sents, train_tgt_sents = load_data(
        os.path.join(DATA_DIR, DATA_FILES["train"][0]),
        os.path.join(DATA_DIR, DATA_FILES["train"][1]),
        NUM_EXAMPLES
    )

    train_src_tensor, src_tokenizer = tokenize_sentences(train_src_sents)
    train_tgt_tensor, tgt_tokenizer = tokenize_sentences(train_tgt_sents)

    vocab_src_size = len(src_tokenizer.word_index) + 1
    vocab_tgt_size = len(tgt_tokenizer.word_index) + 1
    max_len_src = train_src_tensor.shape[1]
    max_len_tgt = train_tgt_tensor.shape[1]

    # --- 2. Create tf.data.Dataset ---
    dataset = tf.data.Dataset.from_tensor_slices((train_src_tensor, train_tgt_tensor))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    steps_per_epoch = len(train_src_tensor) // BATCH_SIZE

    # --- 3. Initialize Model ---
    encoder = Encoder(vocab_src_size, EMBEDDING_DIM, HIDDEN_UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tgt_size, EMBEDDING_DIM, HIDDEN_UNITS, BATCH_SIZE)

    # --- 4. Training Loop ---
    checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for batch, (src, tgt) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(src, tgt, enc_hidden, encoder, decoder, tgt_tokenizer)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}')
        
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Epoch {epoch + 1} Loss {total_loss / steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    # --- 5. Evaluate and Visualize ---
    # Restore the latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    
    test_sentence = "How are you ?"
    result, sentence, attention_plot = evaluate(
        test_sentence, encoder, decoder, src_tokenizer, tgt_tokenizer, max_len_src, max_len_tgt
    )
    
    print(f'Input: {sentence}')
    print(f'Predicted translation: {result}')

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))


if __name__ == '__main__':
    main()