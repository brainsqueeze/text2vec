from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, initializers


class ScaledDotAttention(layers.Layer):
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
        self.neg_inf = tf.constant(-1e9, dtype=tf.float32)

    # pylint: disable=missing-function-docstring
    def call(self, query, key, value, mask_future: bool = False):
        numerator = tf.matmul(query, key, transpose_b=True)
        denominator = tf.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))

        if mask_future:
            upper = tf.linalg.band_part(tf.ones(tf.shape(numerator)[1:], dtype=tf.float32), num_lower=0, num_upper=-1)
            diag = tf.linalg.band_part(upper, num_lower=0, num_upper=0)
            numerator += (self.neg_inf * (upper - diag))

        x = tf.nn.softmax(numerator / denominator)
        return tf.matmul(x, value)


class BahdanauAttention(layers.Layer):
    """Layer which computes the Bahdanau attention mechanism either as a self-attention or as
    a encoder-decoder attention.

    If only the `encoded` input is specified then self-attention will be computed.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    size : int
        The dimensionality of the hidden attention weights. This is the same as the word-embedding dimensionality.
    drop_rate : float, optional
        Value between 0 and 1.0, performs dropout on the attention weights, by default 0.

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

    def __init__(self, size: int, drop_rate: float = 0.):
        super().__init__(name="BahdanauAttention")

        self.hidden = layers.Dense(units=size, activation="tanh")
        self.U = tf.Variable(initializers.GlorotUniform()(shape=[size]), name="U", dtype=tf.float32, trainable=True)
        self.dropout = layers.Dropout(drop_rate)

    # pylint: disable=missing-function-docstring
    def call(self, encoded: tf.Tensor, decoded: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        if decoded is None:
            score = tf.math.reduce_sum(self.U * self.hidden(encoded), axis=-1)
            alphas = tf.nn.softmax(score)
            alphas = self.dropout(alphas, training=training)
            x = tf.expand_dims(alphas, axis=-1) * encoded
            return x, tf.math.reduce_sum(x, axis=1)

        score = tf.einsum("ijm,mn,ikn->ijk", encoded, self.hidden.kernel, decoded)
        alphas = tf.nn.softmax(score, axis=1)
        alphas = self.dropout(alphas, training=training)
        alphas = tf.math.reduce_sum(tf.matmul(alphas, encoded, transpose_a=True), axis=-1)
        x = tf.expand_dims(alphas, axis=-1) * decoded
        return x, tf.math.reduce_sum(x, axis=1)


class SingleHeadAttention(layers.Layer):
    """Layer which computes the single-head-attention mechanism as described in
    https://arxiv.org/abs/1706.03762.

    Query, key, value tensors are submitted to the layer as a tuple. Optional future-masking is available.
    The optional `training` flag toggles dropout on/off.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    num_layers : int, optional
        The number of parallel single-head-attention mechanisms, by default 8.
    drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

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

    def __init__(self, emb_dims, num_layers: int = 8, drop_rate: float = 0.):
        super().__init__(name="SingleHeadAttention")
        assert isinstance(num_layers, int) and num_layers > 0

        dims = emb_dims
        key_dims = emb_dims // num_layers
        initializer = tf.keras.initializers.GlorotUniform()

        self.WQ = tf.Variable(initializer(shape=(dims, key_dims)), name="WQ", dtype=tf.float32, trainable=True)
        self.WK = tf.Variable(initializer(shape=(dims, key_dims)), name="WK", dtype=tf.float32, trainable=True)
        self.WV = tf.Variable(initializer(shape=(dims, key_dims)), name="WV", dtype=tf.float32, trainable=True)
        self.dropout = layers.Dropout(drop_rate)
        self.dot_attention = ScaledDotAttention()

    # pylint: disable=missing-function-docstring
    def call(self, inputs, mask_future: bool = False, training: bool = False):
        queries, keys, values = inputs

        queries = self.dropout(queries, training=training)
        keys = self.dropout(keys, training=training)
        values = self.dropout(values, training=training)

        head_queries = tf.tensordot(queries, self.WQ, axes=[-1, 0])
        head_keys = tf.tensordot(keys, self.WK, axes=[-1, 0])
        head_values = tf.tensordot(values, self.WV, axes=[-1, 0])
        return self.dot_attention(query=head_queries, key=head_keys, value=head_values, mask_future=mask_future)


class MultiHeadAttention(layers.Layer):
    """Layer which computes the multi-head-attention mechanism as described in
    https://arxiv.org/abs/1706.03762.

    Query, key, value tensors are submitted to the layer as a tuple. Optional future-masking is available.
    The optional `training` flag toggles dropout on/off.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    num_layers : int, optional
        The number of parallel single-head-attention mechanisms, by default 8.
    drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

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

    def __init__(self, emb_dims: int, num_layers: int = 8, drop_rate: float = 0.):
        super().__init__(name="MultiHeadAttention")
        self.layer_heads = [
            SingleHeadAttention(emb_dims=emb_dims, num_layers=num_layers, drop_rate=drop_rate)
            for _ in range(num_layers)
        ]
        self.dense = layers.Dense(units=emb_dims, use_bias=False)

    # pylint: disable=missing-function-docstring
    def call(self, inputs, mask_future=False, training=False):
        heads = [layer(inputs, mask_future=mask_future, training=training) for layer in self.layer_heads]
        total_head = tf.concat(heads, -1)
        return self.dense(total_head)
