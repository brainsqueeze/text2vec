import tensorflow as tf


class Embed(tf.keras.layers.Layer):

    def __init__(self, vocab_size: int, embedding_size: int, max_sequence_len: int):
        super().__init__()

        self.embeddings = tf.Variable(
            tf.random.uniform([vocab_size, embedding_size], -1.0, 1.0),
            name='embeddings',
            dtype=tf.float32,
            trainable=True
        )
        self.max_len = tf.constant(max_sequence_len)
        self.slicer = tf.keras.layers.Lambda(lambda x: x[:, :max_sequence_len], name="sequence-slice")

    def call(self, token_ids, **kwargs):
        with tf.name_scope("TokenIds"):
            token_ids = self.slicer(token_ids)
            x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, token_ids)
            x = x.to_tensor(0)

            seq_lengths = token_ids.row_lengths()
            time_steps = tf.cast(tf.reduce_max(seq_lengths), dtype=tf.int32)
            mask = tf.sequence_mask(lengths=seq_lengths, maxlen=time_steps, dtype=tf.float32)
            return x, mask, time_steps

    def get_embedding(self, token_ids):
        with tf.name_scope("TokenEmbeddings"):
            return tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, token_ids)


class TokenEmbed(tf.keras.layers.Layer):

    def __init__(self, token_hash: dict, embedding_size: int, max_sequence_len: int, unknown_token: str = '<unk>'):
        super().__init__()

        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=list(token_hash.keys()),
                values=list(token_hash.values())
            ),
            default_value=token_hash.get(unknown_token)
        )
        self.embed_layer = Embed(
            vocab_size=len(token_hash),
            embedding_size=embedding_size,
            max_sequence_len=max_sequence_len
        )

    def call(self, tokens, **kwargs):
        with tf.name_scope("TextInput"):
            hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
            return self.embed_layer(hashed, **kwargs)

    def get_embedding(self, tokens):
        with tf.name_scope("TextToEmbedding"):
            hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
            return self.embed_layer.get_embedding(hashed)

    @property
    def slicer(self):
        return self.embed_layer.slicer

    @property
    def embeddings(self):
        return self.embed_layer.embeddings
