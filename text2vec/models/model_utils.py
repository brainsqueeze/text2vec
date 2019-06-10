import tensorflow as tf


def scalar_dot_product_attention(query, key, value, mask_future=False):
    with tf.variable_scope('scalar-dot-attention'):
        numerator = tf.einsum('ijk,ilk->ijl', query, key)
        denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

        if mask_future:
            upper = (1 + 1e9) * tf.linalg.band_part(tf.ones_like(numerator), num_lower=0, num_upper=-1)
            mask = 1 - upper
            numerator *= mask

        x = tf.nn.softmax(numerator / denominator)
        return tf.einsum('ijk,ikl->ijl', x, value)


def layer_norm_compute(x, epsilon=1e-8, scale=1.0, bias=0):
    with tf.variable_scope('layer-norm'):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias
