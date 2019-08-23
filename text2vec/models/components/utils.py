import tensorflow as tf
import numpy as np


def ragged_tensor_process_mask(tokens, lookup, embeddings, max_sequence_length):
    assert isinstance(tokens, tf.RaggedTensor)
    assert isinstance(lookup, tf.lookup.StaticHashTable)
    assert isinstance(embeddings, tf.Variable)

    epsilon = 1e-8
    emb_dims = embeddings.shape[-1]

    hashed = tf.ragged.map_flat_values(lookup.lookup, tokens)
    hashed = hashed[:, :max_sequence_length]
    batch_size = hashed.nrows()
    seq_lengths = hashed.row_lengths()
    time_steps = tf.cast(tf.reduce_max(seq_lengths), dtype=tf.int32)
    padding = tf.zeros(shape=(batch_size, max_sequence_length - time_steps, emb_dims), dtype=tf.float32)

    x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, embeddings, hashed)
    x = x.to_tensor()
    # pad to full max sequence length, otherwise we get numerical inconsistencies with differing batch sizes
    x = tf.concat([x, padding], axis=1)

    # time-step masking
    dec_mask = tf.sequence_mask(lengths=seq_lengths, maxlen=max_sequence_length)
    dec_mask = tf.cast(dec_mask, dtype=tf.float32)
    dec_mask = tf.tile(tf.expand_dims(dec_mask, axis=-1), multiples=[1, 1, emb_dims]) + epsilon
    return x, dec_mask, time_steps


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
