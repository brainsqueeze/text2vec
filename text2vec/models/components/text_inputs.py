import tensorflow as tf


class Embed(tf.keras.layers.Layer):
    """This layer handles the primary text feature transformations and word-embeddings to be passed off
    to the sequence-aware parts of the encoder/decoder pipeline.

    Texts come in already tokenized, and are already converted to integer index values. The tokens are used to lookup
    word-embeddings and a sequence mask is computed.

    The inputs are `tf.RaggedTensor` types, and after word-embeddings the tensor is made dense by padding to the
    longest sequence length in the batch.

    The `get_embedding` method can be used to get the token embedding vectors.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_size : int
        Dimensionality of the word-embeddings.
    max_sequence_len : int
        Longest sequence seen at training time. This layer ensures that no input sequences exceed this length.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import Embed

    embedder = Embed(vocab_size=4, embedding_size=16, max_sequence_len=10)

    tok_ids = tf.ragged.constant([[3, 0, 3], [3, 1, 3, 3, 2, 3]])
    sequences, seq_mask, time_steps = embedder(tok_ids)

    # get word-embeddings only
    word_embeddings = embedder.get_embedding(tok_ids)
    ```
    """

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
            time_steps = tf.cast(tf.reduce_max(seq_lengths), tf.int32)
            mask = tf.sequence_mask(lengths=seq_lengths, maxlen=time_steps, dtype=tf.float32)
            return x, mask, time_steps

    def get_embedding(self, token_ids: tf.RaggedTensor) -> tf.RaggedTensor:
        """Get the token embeddings for the input IDs.

        Parameters
        ----------
        token_ids : tf.RaggedTensor
            Sequences of token IDs

        Returns
        -------
        tf.RaggedTensor
            Sequences of token embeddings with the same number of time steps as `token_ids`
        """

        with tf.name_scope("TokenEmbeddings"):
            return tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, token_ids)


class TokenEmbed(tf.keras.layers.Layer):
    """This layer handles the primary text feature transformations and word-embeddings to be passed off
    to the sequence-aware parts of the encoder/decoder pipeline.

    Texts come in already tokenized. The tokens are transformed to integer index values from a
    `tf.lookup.StaticHashTable` lookup table. The tokens are used to lookup word-embeddings and a sequence
    mask is computed.

    The inputs are `tf.RaggedTensor` types, and after word-embeddings the tensor is made dense by padding to the
    longest sequence length in the batch.

    The `get_embedding` method can be used to get the token embedding vectors.

    Parameters
    ----------
    token_hash : dict
        Token -> integer vocabulary lookup.
    embedding_size : int
        Dimensionality of the word-embeddings.
    max_sequence_len : int
        Longest sequence seen at training time. This layer ensures that no input sequences exceed this length.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TokenEmbed

    lookup = {'string': 0, 'is': 1, 'example': 2, '<unk>': 3}
    embedder = TokenEmbed(token_hash=lookup, embedding_size=16, max_sequence_len=10)

    text = tf.ragged.constant([["sample", "string", "."], ["this", "is", "a", "second", "example", "."]])
    sequences, seq_mask, time_steps = embedder(text)

    # get word-embeddings only
    word_embeddings = embedder.get_embedding(text)
    ```
    """

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

    def get_embedding(self, tokens: tf.RaggedTensor) -> tf.RaggedTensor:
        """Get the token embeddings for the input tokens.

        Parameters
        ----------
        tokens : tf.RaggedTensor
            Sequences of tokens

        Returns
        -------
        tf.RaggedTensor
            Sequences of token embeddings with the same number of time steps as `tokens`
        """

        with tf.name_scope("TextToEmbedding"):
            hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
            return self.embed_layer.get_embedding(hashed)

    @property
    def slicer(self):
        return self.embed_layer.slicer

    @property
    def embeddings(self):
        return self.embed_layer.embeddings
