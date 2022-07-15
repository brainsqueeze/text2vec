import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers


class LayerNorm(layers.Layer):
    """Layer normalization, independent of batch size.

    Parameters
    ----------
    epsilon : float, optional
        Regularization parameter when dividing by the variance, by default 1e-8
    scale : float, optional
        Scaling factor for the centralized and normalized values, by default 1.0
    bias : float, optional
        Bias offset value, by default 0

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import utils

    normalizer = utils.LayerNorm()
    x = tf.random.uniform(shape=[4, 7, 12])
    normalizer(x)
    ```
    """

    def __init__(self, epsilon: float = 1e-8, scale: float = 1.0, bias: float = 0):
        super().__init__()
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.scale = tf.constant(scale, dtype=tf.float32)
        self.bias = tf.constant(bias, dtype=tf.float32)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
        return norm * self.scale + self.bias


class TensorProjection(layers.Layer):
    """Projects sequence vectors onto a fixed vector. This returns a new tensor with the same shape as the
    input tensor, with all sequence vectors projected.

    The vectors for projection should have the same dimensions as the input tensor `x`, `[batch_size, T, dim]`.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import utils

    projector = utils.TensorProjection()
    x = tf.random.uniform(shape=[4, 7, 12])
    vectors = tf.random.uniform(shape=[4, 12])
    projected_tensor = projector(x, projection_vector=vectors)
    ```
    """

    def __init__(self):
        super().__init__()

    def call(self, x, projection_vector):
        projection_vector = tf.math.l2_normalize(projection_vector, axis=-1)
        inner_product = tf.einsum("ijk,ik->ij", x, projection_vector)
        return tf.einsum("ij,ik->ijk", inner_product, projection_vector)


class PositionalEncoder(layers.Layer):
    """Layer which initializes the positional encoding tensor, and defines the operation which adds the encoding
    to an input tensor and then applies a sequence mask.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    max_sequence_len : int
        Longest sequence seen at training time.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TokenEmbed
    from text2vec.models import utils

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TokenEmbed(token_hash=lookup, embedding_size=16, max_sequence_len=10)
    encoder = utils.PositionalEncoder(emb_dims=16, max_sequence_len=10)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    x, mask, _ = inputer(text)
    encoder(x, mask)
    ```
    """

    def __init__(self, emb_dims: int, max_sequence_len: int):
        super().__init__()

        positions = np.arange(max_sequence_len).astype(np.float32)
        column_range = np.arange(emb_dims).astype(np.float32)
        factor = np.power(1e5 ** (2 / emb_dims), column_range)

        even = np.sin(positions / factor[::2, None]).T
        odd = np.cos(positions / factor[1::2, None]).T

        encoder = np.zeros(shape=(max_sequence_len, emb_dims), dtype=np.float32)
        encoder[:, ::2] = even
        encoder[:, 1::2] = odd
        self.encoder = tf.convert_to_tensor(encoder, dtype=tf.float32)

    def call(self, x: tf.Tensor, mask: tf.Tensor):
        time_steps = tf.shape(x)[1]
        return tf.expand_dims(mask, axis=-1) * (x + self.encoder[:time_steps, ...])


class VariationPositionalEncoder(layers.Layer):
    """Learns the relative phases between sequence steps in an attention-based transformer, where there is no
    inherent sequential ordering.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.
    max_sequence_len : int
        Longest sequence seen at training time.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TokenEmbed
    from text2vec.models import utils

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TokenEmbed(token_hash=lookup, embedding_size=16, max_sequence_len=10)
    encoder = utils.VariationPositionalEncoder(emb_dims=16, max_sequence_len=10)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    x, mask, _ = inputer(text)
    encoder(x, mask)
    ```
    """

    def __init__(self, emb_dims: int, max_sequence_len: int):
        super().__init__()

        self.encoder = tf.Variable(
            initializers.GlorotUniform()(shape=(max_sequence_len, emb_dims)),
            dtype=tf.float32,
            trainable=True,
            name="positional-encoder"
        )

    def call(self, x: tf.Tensor, mask: tf.Tensor):
        time_steps = tf.shape(x)[1]
        return tf.expand_dims(mask, axis=-1) * (x + self.encoder[:time_steps, ...])
