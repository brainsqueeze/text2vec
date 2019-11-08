import tensorflow as tf


class Tokenizer(tf.keras.layers.Layer):

    def __init__(self, sep=' '):
        super(Tokenizer, self).__init__(name="Tokenizer")
        self.sep = sep

    def call(self, corpus):
        return tf.strings.split(corpus, self.sep)


class TextInput(tf.keras.layers.Layer):

    def __init__(self, token_hash, embedding_size, max_sequence_len, epsilon=1e-8, **kwargs):
        super(TextInput, self).__init__()
        assert isinstance(token_hash, dict)

        self.epsilon = tf.constant(epsilon)
        self.num_labels = tf.constant(len(token_hash) + 1)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(list(token_hash.keys()), list(token_hash.values())),
            default_value=max(token_hash.values()) + 1
        )
        self.embeddings = tf.Variable(
            tf.random.uniform([self.num_labels, embedding_size], -1.0, 1.0),
            name='embeddings',
            dtype=tf.float32,
            trainable=True
        )
        self.max_len = tf.constant(max_sequence_len)
        self.slicer = tf.keras.layers.Lambda(lambda x: x[:, :max_sequence_len])

    def call(self, tokens):
        emb_dims = self.embeddings.shape[-1]

        hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
        hashed = self.slicer(hashed)
        batch_size = hashed.nrows()
        seq_lengths = hashed.row_lengths()
        time_steps = tf.cast(tf.reduce_max(seq_lengths), dtype=tf.int32)
        padding = tf.zeros(shape=(batch_size, self.max_len - time_steps, emb_dims), dtype=tf.float32)

        x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, hashed)
        x = x.to_tensor()

        # pad to full max sequence length
        # otherwise we get numerical inconsistencies with differing batch sizes
        x = tf.concat([x, padding], axis=1)

        # time-step masking
        dec_mask = tf.sequence_mask(lengths=seq_lengths, maxlen=self.max_len)
        dec_mask = tf.cast(dec_mask, dtype=tf.float32)
        dec_mask = tf.tile(tf.expand_dims(dec_mask, axis=-1), multiples=[1, 1, emb_dims]) + self.epsilon
        return x, dec_mask, time_steps
