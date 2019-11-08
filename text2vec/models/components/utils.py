import tensorflow as tf
import numpy as np


class ScalarDotAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(ScalarDotAttention, self).__init__(name="ScalarDotAttention")

    def __call__(self, query, key, value, mask_future=False):
        with tf.name_scope("ScalarDotAttention"):
            numerator = tf.einsum('ijk,ilk->ijl', query, key)
            denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

            if mask_future:
                upper = (1 + 1e9) * tf.linalg.band_part(tf.ones_like(numerator), num_lower=0, num_upper=-1)
                mask = 1 - upper
                numerator *= mask

            x = tf.nn.softmax(numerator / denominator)
            return tf.einsum('ijk,ikl->ijl', x, value)


class LayerNorm(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-8, scale=1.0, bias=0):
        super(LayerNorm, self).__init__(name="LayerNorm")
        self.epsilon = tf.constant(epsilon, dtype=tf.float32)
        self.scale = tf.constant(scale, dtype=tf.float32)
        self.bias = tf.constant(bias, dtype=tf.float32)

    def __call__(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
        return norm * self.scale + self.bias


class TensorProjection(tf.keras.layers.Layer):

    def __init__(self):
        super(TensorProjection, self).__init__(name="TensorProjection")

    def __call__(self, x, projection_vector):
        inner_product = tf.einsum("ijk,ik->ij", x, projection_vector)
        time_steps = tf.shape(x)[1]
        p_vector_norm_squared = tf.norm(projection_vector, axis=1) ** 2
        p_vector_norm_squared = tf.tile(tf.expand_dims(p_vector_norm_squared, -1), [1, time_steps])

        alpha = tf.divide(inner_product, p_vector_norm_squared)
        return tf.einsum("ij,ik->ijk", alpha, projection_vector)


def positional_encode(emb_dims, max_sequence_length):
    with tf.name_scope('PositionalEncoder'):
        positions = np.arange(max_sequence_length).astype(np.float32)
        column_range = np.arange(emb_dims).astype(np.float32)
        factor = np.power(1e5 ** (2 / emb_dims), column_range)

        even = np.sin(positions / factor[::2, None]).T
        odd = np.cos(positions / factor[1::2, None]).T

        encoder = np.zeros(shape=(max_sequence_length, emb_dims), dtype=np.float32)
        encoder[:, ::2] = even
        encoder[:, 1::2] = odd
        encoder = tf.convert_to_tensor(encoder, dtype=tf.float32)
        return encoder
