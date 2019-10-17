import tensorflow as tf
import numpy as np


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


def tensor_projection(x, p_vector):
    assert isinstance(x, tf.Tensor)

    with tf.name_scope('projection'):
        inner_product = tf.einsum("ijk,ik->ij", x, p_vector)
        time_steps = tf.shape(x)[1]
        p_vector_norm_squared = tf.norm(p_vector, axis=1) ** 2
        p_vector_norm_squared = tf.tile(tf.expand_dims(p_vector_norm_squared, -1), [1, time_steps])

        alpha = tf.divide(inner_product, p_vector_norm_squared)
        return tf.einsum("ij,ik->ijk", alpha, p_vector)


def positional_encode(emb_dims, max_sequence_length):
    with tf.name_scope('positional-encoder'):
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


def sequence_cost(target_sequences, sequence_logits, num_labels, smoothing=False):
    with tf.name_scope('cost'):
        if smoothing:
            smoothing = 0.1
            targets = tf.one_hot(target_sequences, depth=num_labels, on_value=1.0, off_value=0.0, axis=-1)
            loss = tf.losses.softmax_cross_entropy(
                logits=sequence_logits,
                onehot_labels=targets,
                label_smoothing=smoothing,
                reduction=tf.losses.Reduction.NONE
            )
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=sequence_logits,
                labels=target_sequences
            )

        loss = tf.reduce_mean(loss)
        return loss
