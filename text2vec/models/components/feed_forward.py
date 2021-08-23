import tensorflow as tf


class PositionWiseFFN(tf.keras.layers.Layer):
    """Position-wise feed-forward network implemented as conv -> relu -> conv.
    1D convolutions of the input tensor are computed to an intermediate hidden dimension, then a final 1D
    convolution is computed of the ReLu output from the intermediate layer to return to the original input shape.

    Input tensor shapes should be `[batch_size, T, dim]`.

    Parameters
    ----------
    emb_dims : int
        The word-embedding dimensionality. This value determines the dimensionalities of the hidden weights.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import PositionWiseFFN

    X = tf.random.uniform(shape=[4, 7, 12])
    ffn = PositionWiseFFN(emb_dims=12)
    ffn(X)
    ```
    """

    def __init__(self, emb_dims):
        super().__init__()

        self.conv_inner = tf.keras.layers.Conv1D(
            filters=4 * emb_dims,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation='relu'
        )
        self.conv_outer = tf.keras.layers.Conv1D(filters=emb_dims, kernel_size=1, padding='same', use_bias=False)

    def call(self, x):
        with tf.name_scope("PositionWiseFFN"):
            return self.conv_outer(self.conv_inner(x))
