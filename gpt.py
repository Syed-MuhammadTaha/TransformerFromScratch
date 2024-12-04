import tensorflow as tf
from tensorflow.keras import layers, Model

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = None  # Will be defined based on the input data
# ------------

# Load and preprocess data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = tf.constant(encode(text), dtype=tf.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(data, split):
    idx = tf.random.uniform([batch_size], maxval=len(data) - block_size, dtype=tf.int32)
    x = tf.stack([data[i:i + block_size] for i in idx])
    y = tf.stack([data[i + 1:i + block_size + 1] for i in idx])
    return x, y

# Model components
class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.all_head_size = num_heads * head_size

        self.query = layers.Dense(self.all_head_size)
        self.key = layers.Dense(self.all_head_size)
        self.value = layers.Dense(self.all_head_size)
        self.out = layers.Dense(n_embd)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        B, T, C = tf.shape(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = tf.reshape(q, (B, T, self.num_heads, self.head_size))
        k = tf.reshape(k, (B, T, self.num_heads, self.head_size))
        v = tf.reshape(v, (B, T, self.num_heads, self.head_size))

        # Scaled dot-product attention
        scores = tf.einsum('bthd,bThd->bhtT', q, k) / tf.sqrt(float(self.head_size))
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)  # Lower triangular matrix
        scores = tf.where(mask == 0, -float('inf'), scores)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        context = tf.einsum('bhtT,bThd->bthd', weights, v)

        # Concatenate heads and project
        context = tf.reshape(context, (B, T, -1))
        return self.out(context)

class FeedForward(layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(4 * n_embd, activation='relu'),
            layers.Dense(n_embd),
            layers.Dropout(dropout),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)

class TransformerBlock(layers.Layer):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.mha = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x, training=False):
        x = x + self.mha(self.ln1(x), training=training)
        x = x + self.ffwd(self.ln2(x), training=training)
        return x

class GPTLanguageModel(Model):
    def __init__(self):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, n_embd)
        self.position_embedding = layers.Embedding(block_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, x, targets=None, training=False):
        B, T = tf.shape(x)
        tok_emb = self.token_embedding(x)  # (B, T, C)
        pos_emb = self.position_embedding(tf.range(T))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is not None:
            targets = tf.reshape(targets, [-1])
            logits = tf.reshape(logits, [-1, vocab_size])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
            return logits, tf.reduce_mean(loss)
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self.call(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = tf.nn.softmax(logits, axis=-1)
            next_idx = tf.random.categorical(probs, num_samples=1)
            idx = tf.concat([idx, next_idx], axis=1)
        return idx

# Training
model = GPTLanguageModel()
optimizer = tf.keras.optimizers.Adam(learning_rate)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        _, loss = model(x, targets=y, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for iter in range(max_iters):
    xb, yb = get_batch(train_data, 'train')
    loss = train_step(xb, yb)

    if iter % eval_interval == 0 or iter == max_iters - 1:
        print(f"Step {iter}: Loss = {loss.numpy()}")

# Generate
context = tf.zeros((1, 1), dtype=tf.int32)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated.numpy().tolist()[0]))
