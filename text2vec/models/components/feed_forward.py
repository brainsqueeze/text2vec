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
        hidden_dim_size = 4 * emb_dims

        self.ConvInner = tf.Variable(
            tf.zeros([1, emb_dims, hidden_dim_size]),
            name='conv-filter-inner',
            dtype=tf.float32,
            trainable=True
        )
        self.ConvOuter = tf.Variable(
            tf.zeros([1, hidden_dim_size, emb_dims]),
            name='conv-filter-outer',
            dtype=tf.float32,
            trainable=True
        )

    def call(self, x):
        with tf.name_scope("PositionWiseFFN"):
            x = tf.nn.conv1d(x, filters=self.ConvInner, stride=1, padding='SAME')
            x = tf.nn.relu(x)
            return tf.nn.conv1d(x, filters=self.ConvOuter, stride=1, padding='SAME')
