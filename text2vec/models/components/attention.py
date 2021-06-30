import tensorflow as tf


class ScaledDotAttention(tf.keras.layers.Layer):
    """Scaled dot attention layer which computes
    ```
    softmax(Query * permutedim(Key, (3, 1, 2)) / sqrt(dk)) * permutedim(Value, (2, 1, 3))
    ```
    where `dk` is the dimensionality of each sequence in the Key tensor.

    Future sequence-step masking can be performed, optionally, where the upper-triangular part of the softmax
    output tensor is set to 0 values.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models.components.attention import ScaledDotAttention

    Q = tf.random.uniform(shape=[4, 7, 12])
    K = tf.random.uniform(shape=[4, 7, 12])
    V = tf.random.uniform(shape=[4, 7, 12])

    attention = ScaledDotAttention()

    # un-masked scalar dot attention
    attention(Q, K, V)

    # future masking
    attention(Q, K, V, mask_future=True)
    ```
    """

    def __init__(self):
        super().__init__(name="ScaledDotAttention")

    def call(self, query, key, value, mask_future=False):
        with tf.name_scope("ScaledDotAttention"):
            numerator = tf.einsum('ijk,ilk->ijl', query, key)
            denominator = tf.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))

            if mask_future:
                upper = (1 + 1e9) * tf.linalg.band_part(tf.ones_like(numerator), num_lower=0, num_upper=-1)
                mask = 1 - upper
                numerator *= mask

            x = tf.nn.softmax(numerator / denominator)
            return tf.einsum('ijk,ikl->ijl', x, value)


class BahdanauAttention(tf.keras.layers.Layer):
    """Layer which computes the Bahdanau attention mechanism either as a self-attention or as
    a encoder-decoder attention.

    If only the `encoded` input is specified then self-attention will be computed.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    size : int
        The dimensionality of the hidden attention weights. This is the same as the word-embedding dimensionality.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import BahdanauAttention

    dims = 12
    encoded_sequences = tf.random.uniform(shape=[4, 7, dims])
    decoded_sequences = tf.random.uniform(shape=[4, 11, dims])
    attention = BahdanauAttention(dims)

    # self attention
    attention(encoded_sequences)

    # mutual attention
    attention(encoded_sequences, decoded_sequences)
    ```
    """

    def __init__(self, size):
        super().__init__(name="BahdanauAttention")

        initializer = tf.keras.initializers.GlorotUniform()
        self.W = tf.Variable(
            initializer(shape=(size, size)),
            name='weight',
            dtype=tf.float32,
            trainable=True
        )
        self.B = tf.Variable(tf.zeros(shape=[size]), name="B", dtype=tf.float32, trainable=True)
        self.U = tf.Variable(initializer(shape=[size]), name="U", dtype=tf.float32, trainable=True)

    def call(self, encoded, decoded=None):
        with tf.name_scope("BahdanauAttention"):
            if decoded is None:
                score = tf.math.tanh(tf.tensordot(encoded, self.W, axes=[-1, 0]) + self.B)
                score = tf.reduce_sum(self.U * score, axis=-1)
                alphas = tf.nn.softmax(score, name="attention-weights")
                # encoded = encoded * tf.expand_dims(alphas, axis=-1)
                # return encoded, tf.reduce_sum(encoded, axis=1, name="context-vector")
                return tf.einsum('ilk,il->ik', encoded, alphas)

            score = tf.einsum("ijm,mn,ikn->ijk", encoded, self.W, decoded)
            alphas = tf.reduce_mean(score, axis=1)
            alphas = tf.nn.softmax(alphas)
            # decoded = decoded * tf.expand_dims(alphas, axis=-1)
            # return decoded, tf.reduce_sum(decoded, axis=1)
            return tf.einsum('ilk,il->ik', decoded, alphas)


class SingleHeadAttention(tf.keras.layers.Layer):
    """Layer which computes the single-head-attention mechanism as described in
    https://arxiv.org/abs/1706.03762.

    Query, key, value tensors are submitted to the layer as a tuple. Optional future-masking is available.
    The optional `training` flag toggles dropout on/off.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    layers : int, optional
        The number of parallel single-head-attention mechanisms, by default 8.
    keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models.components.attention import SingleHeadAttention

    Q = tf.random.uniform(shape=[4, 7, 12])
    K = tf.random.uniform(shape=[4, 11, 12])
    V = tf.random.uniform(shape=[4, 5, 12])

    # 25% dropout rate
    attention = SingleHeadAttention(emb_dims=12, keep_prob=0.75)

    # masking and dropout turned on
    attention(inputs=(Q, K, V), mask_future=True, training=True)
    ```
    """

    def __init__(self, emb_dims, layers=8, keep_prob=1.0):
        super().__init__(name="SingleHeadAttention")
        assert isinstance(layers, int) and layers > 0

        dims = emb_dims
        key_dims = emb_dims // layers
        initializer = tf.keras.initializers.GlorotUniform()

        self.WQ = tf.Variable(initializer(shape=(dims, key_dims)), name="WQ", dtype=tf.float32, trainable=True)
        self.WK = tf.Variable(initializer(shape=(dims, key_dims)), name="WK", dtype=tf.float32, trainable=True)
        self.WV = tf.Variable(initializer(shape=(dims, key_dims)), name="WV", dtype=tf.float32, trainable=True)
        self.dropout = tf.keras.layers.Dropout(1 - keep_prob)
        self.dot_attention = ScaledDotAttention()

    def call(self, inputs, mask_future=False, training=False):
        with tf.name_scope("SingleHeadAttention"):
            queries, keys, values = inputs

            queries = self.dropout(queries, training=training)
            keys = self.dropout(keys, training=training)
            values = self.dropout(values, training=training)

            head_queries = tf.tensordot(queries, self.WQ, axes=[-1, 0])
            head_keys = tf.tensordot(keys, self.WK, axes=[-1, 0])
            head_values = tf.tensordot(values, self.WV, axes=[-1, 0])
            return self.dot_attention(query=head_queries, key=head_keys, value=head_values, mask_future=mask_future)


class MultiHeadAttention(tf.keras.layers.Layer):
    """Layer which computes the multi-head-attention mechanism as described in
    https://arxiv.org/abs/1706.03762.

    Query, key, value tensors are submitted to the layer as a tuple. Optional future-masking is available.
    The optional `training` flag toggles dropout on/off.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    layers : int, optional
        The number of parallel single-head-attention mechanisms, by default 8.
    keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import MultiHeadAttention

    Q = tf.random.uniform(shape=[4, 7, 12])
    K = tf.random.uniform(shape=[4, 11, 12])
    V = tf.random.uniform(shape=[4, 5, 12])

    # 25% dropout rate
    attention = MultiHeadAttention(emb_dims=12, keep_prob=0.75)

    # masking and dropout turned on
    attention(inputs=(Q, K, V), mask_future=True, training=True)
    ```
    """

    def __init__(self, emb_dims, layers=8, keep_prob=1.0):
        super().__init__(name="MultiHeadAttention")
        self.layer_heads = []
        for i in range(layers):
            with tf.name_scope(f"head-{i}"):
                self.layer_heads.append(SingleHeadAttention(emb_dims=emb_dims, layers=layers, keep_prob=keep_prob))

        self.dense = tf.keras.layers.Dense(units=emb_dims, use_bias=False)

    def call(self, inputs, mask_future=False, training=False):
        with tf.name_scope("MultiHeadAttention"):
            heads = [layer(inputs, mask_future=mask_future, training=training) for layer in self.layer_heads]
            total_head = tf.concat(heads, -1)
            return self.dense(total_head)
