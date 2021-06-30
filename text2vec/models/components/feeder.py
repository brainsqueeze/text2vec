import tensorflow as tf


class Tokenizer(tf.keras.layers.Layer):
    """String-splitting layer.

    Parameters
    ----------
    sep : str, optional
        The token to split the incoming strings by, by default ' '.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import Tokenizer

    text = tf.constant([
        "Sample string.",
        "This is a second example."
    ])
    tokenizer = Tokenizer()
    tokenizer(text)
    ```
    """

    def __init__(self, sep=' '):
        super().__init__(name="Tokenizer")
        self.sep = sep

    def call(self, corpus):
        return tf.strings.split(corpus, self.sep)


class TextInput(tf.keras.layers.Layer):
    """This layer handles the primary text feature transformations and word-embeddings to be passed off
    to the sequence-aware parts of the encoder/decoder pipeline.

    Texts come in already tokenized. The tokens are transformed to integer index values from a
    `tf.lookup.StaticHashTable` lookup table. The tokens are used to lookup word-embeddings and a sequence
    mask is computed.

    The inputs are `tf.RaggedTensor` types, and after word-embeddings the tensor is made dense by padding to the
    longest sequence length in the batch.

    In certain cases, only the word-embedding output is necessary, in which case `output_embeddings` can be set `True`
    in the `__call__` method. This by-passes the padding and sequence masking steps.

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
    from text2vec.models import TextInput

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TextInput(token_hash=lookup, embedding_size=16, max_sequence_len=10)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    sequences, seq_mask, time_steps = inputer(text)

    # get word-embeddings only
    word_embeddings = inputer(text, output_embeddings=True)
    ```
    """

    def __init__(self, token_hash, embedding_size, max_sequence_len):
        super().__init__()
        assert isinstance(token_hash, dict)

        self.num_labels = tf.constant(len(token_hash) + 1)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=list(token_hash.keys()),
                values=list(token_hash.values()),
                value_dtype=tf.int32
            ),
            default_value=max(token_hash.values()) + 1
        )
        self.embeddings = tf.Variable(
            tf.random.uniform([self.num_labels, embedding_size], -1.0, 1.0),
            name='embeddings',
            dtype=tf.float32,
            trainable=True
        )
        self.max_len = tf.constant(max_sequence_len)
        self.slicer = tf.keras.layers.Lambda(lambda x: x[:, :max_sequence_len], name="sequence-slice")

    def call(self, tokens, output_embeddings=False):
        with tf.name_scope("TextInput"):
            hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
            hashed = self.slicer(hashed)

            x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, hashed)
            if output_embeddings:
                return x

            x = x.to_tensor(0)
            x = x * tf.math.sqrt(tf.cast(tf.shape(self.embeddings)[-1], tf.float32))  # sqrt(embedding_size)

            seq_lengths = hashed.row_lengths()
            time_steps = tf.cast(tf.reduce_max(seq_lengths), tf.int32)
            mask = tf.sequence_mask(lengths=seq_lengths, maxlen=time_steps, dtype=tf.float32)
            return x, mask, time_steps
