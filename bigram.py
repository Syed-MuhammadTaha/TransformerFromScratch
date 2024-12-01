import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 0.01
eval_iters = 200
#----------------------------------------

with open('input.txt') as f:
    text = f.read()

chars = list(set(text))
vocab_size = len(chars)

encoder = {ch:i for i,ch in enumerate(chars)}
decoder = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [encoder[c] for c in s]
decode = lambda l: ''.join([decoder[i] for i in l])

data = tf.convert_to_tensor(encode(text), dtype=tf.int32)
train_size = int(len(data) * 0.9)
train_data = data[:train_size]
val_data = data[train_size:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = tf.random.uniform((batch_size,), minval=0, maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([data[i:i+block_size] for i in ix])
    y = tf.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(Model):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # Embedding layer: Maps tokens to logits
        self.token_embedding_table = Embedding(vocab_size, vocab_size)

    def call(self, idx, targets=None):
        """
        Forward pass for training and inference.
        - idx: Input token indices (batch_size, seq_length)
        - targets: Target token indices (optional for training)
        """
        # (batch_size, seq_length, vocab_size)
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            # Reshape logits and targets for loss computation
            batch_size, seq_length, vocab_size = logits.shape
            logits = tf.reshape(logits, (batch_size * seq_length, vocab_size))
            targets = tf.reshape(targets, (batch_size * seq_length,))
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)

        return logits, tf.reduce_mean(loss) if loss is not None else None

    def generate(self, idx, max_new_tokens):
        """
        Generate a sequence of tokens given a starting context.
        - idx: Input token indices (batch_size, seq_length)
        - max_new_tokens: Number of tokens to generate
        """
        for _ in range(max_new_tokens):
            logits, _ = self.call(idx)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1)
            idx = tf.concat([idx, idx_next], axis=1)

        return idx
    
model = BigramLanguageModel(vocab_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)

model = BigramLanguageModel(vocab_size)

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

for epoch in range(max_iters):
    xb, yb = get_batch('train')  # Get a batch of input and target tokens
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)  # Forward pass
    gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update weights

    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

start_idx = tf.zeros((1, 1), dtype=tf.int64)
generated_tokens = model.generate(start_idx, max_new_tokens=1000)
print("Generated tokens:", generated_tokens.numpy())

print(decode(generated_tokens[0].numpy()))