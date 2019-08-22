import tensorflow as tf


def scalar_dot_product_attention(query, key, value, mask_future=False):
    with tf.name_scope('scalar-dot-attention'):
        numerator = tf.einsum('ijk,ilk->ijl', query, key)
        denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

        if mask_future:
            upper = (1 + 1e9) * tf.linalg.band_part(tf.ones_like(numerator), num_lower=0, num_upper=-1)
            mask = 1 - upper
            numerator *= mask

        x = tf.nn.softmax(numerator / denominator)
        return tf.einsum('ijk,ikl->ijl', x, value)


def layer_norm_compute(x, epsilon=1e-8, scale=1.0, bias=0):
    with tf.name_scope('layer-norm'):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return norm_x * scale + bias


class Attention(object):

    def __init__(self, size):
        self.weight = tf.Variable(
            tf.random.truncated_normal([size, size], mean=-0.01, stddev=0.01),
            name='weight',
            dtype=tf.float32,
            trainable=True
        )

        self.b_omega = tf.Variable(tf.zeros(shape=[size]), name="b_omega", dtype=tf.float32, trainable=True)
        self.u_omega = tf.Variable(tf.zeros(shape=[size]), name="u_omega", dtype=tf.float32, trainable=True)

    def __call__(self, encoded, decoded=None):
        with tf.name_scope('bahdanau-attention'):
            if decoded is None:
                score = tf.tanh(tf.tensordot(encoded, self.weight, axes=[-1, 0]) + self.b_omega)
                score = tf.reduce_sum(self.u_omega * score, axis=-1)
                alphas = tf.nn.softmax(score, name="attention-weights")
                return tf.reduce_sum(encoded * tf.expand_dims(alphas, -1), 1, name="context-vector")
            else:
                score = tf.einsum("ijm,mn,ikn->ijk", encoded, self.weight, decoded)
                alphas = tf.nn.softmax(score)
                alphas = tf.reduce_sum(alphas, axis=1)
                return tf.reduce_sum(decoded * tf.expand_dims(alphas, -1), axis=1)
