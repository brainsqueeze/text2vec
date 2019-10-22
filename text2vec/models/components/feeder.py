import tensorflow as tf


class InputFeeder(tf.keras.layers.Layer):

    def __init__(self, token_hash, emb_dims):
        super(InputFeeder, self).__init__()
        assert isinstance(token_hash, dict)

        self.num_labels = len(token_hash) + 1
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(list(token_hash.keys()), list(token_hash.values())),
            default_value=max(token_hash.values()) + 1
        )
        self.embeddings = tf.Variable(
            tf.random.uniform([self.num_labels, emb_dims], -1.0, 1.0),
            name='embeddings',
            dtype=tf.float32,
            trainable=True
        )

    def token_lookup(self, x):
        assert isinstance(x, tf.RaggedTensor)
        return tf.ragged.map_flat_values(self.table.lookup, x)

    def ragged_tensor_process_mask(self, tokens, max_sequence_length):
        assert isinstance(tokens, tf.RaggedTensor)

        epsilon = 1e-8
        emb_dims = self.embeddings.shape[-1]

        hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
        hashed = hashed[:, :max_sequence_length]
        batch_size = hashed.nrows()
        seq_lengths = hashed.row_lengths()
        time_steps = tf.cast(tf.reduce_max(seq_lengths), dtype=tf.int32)
        padding = tf.zeros(shape=(batch_size, max_sequence_length - time_steps, emb_dims), dtype=tf.float32)

        x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, hashed)
        x = x.to_tensor()
        # pad to full max sequence length, otherwise we get numerical inconsistencies with differing batch sizes
        x = tf.concat([x, padding], axis=1)

        # time-step masking
        dec_mask = tf.sequence_mask(lengths=seq_lengths, maxlen=max_sequence_length)
        dec_mask = tf.cast(dec_mask, dtype=tf.float32)
        dec_mask = tf.tile(tf.expand_dims(dec_mask, axis=-1), multiples=[1, 1, emb_dims]) + epsilon
        return x, dec_mask, time_steps

    def __call__(self, inputs, max_sequence_length=0, **kwargs):
        assert isinstance(inputs, tf.RaggedTensor)
        return self.ragged_tensor_process_mask(tokens=inputs, max_sequence_length=max_sequence_length)
