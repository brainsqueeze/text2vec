import tensorflow as tf
from .utils import scalar_dot_product_attention


class BahdanauAttention(object):

    def __init__(self, size):
        self.W = tf.Variable(
            tf.random.truncated_normal([size, size], mean=-0.01, stddev=0.01),
            name='weight',
            dtype=tf.float32,
            trainable=True
        )
        self.B = tf.Variable(tf.zeros(shape=[size]), name="B", dtype=tf.float32, trainable=True)
        self.U = tf.Variable(tf.zeros(shape=[size]), name="U", dtype=tf.float32, trainable=True)

    def __call__(self, encoded, decoded=None):
        with tf.name_scope('bahdanau-attention'):
            if decoded is None:
                score = tf.tanh(tf.tensordot(encoded, self.W, axes=[-1, 0]) + self.B)
                score = tf.reduce_sum(self.U * score, axis=-1)
                alphas = tf.nn.softmax(score, name="attention-weights")
                return tf.reduce_sum(encoded * tf.expand_dims(alphas, -1), 1, name="context-vector")
            else:
                score = tf.einsum("ijm,mn,ikn->ijk", encoded, self.W, decoded)
                alphas = tf.nn.softmax(score)
                alphas = tf.reduce_sum(alphas, axis=1)
                return tf.reduce_sum(decoded * tf.expand_dims(alphas, -1), axis=1)


class SingleHeadAttention(object):

    def __init__(self, emb_dims, layers=8):
        assert isinstance(layers, int) and layers > 0

        self.dims = emb_dims
        self.key_dims = emb_dims // layers

        self.WQ = tf.Variable(self.__kernel(), name="WQ", dtype=tf.float32, trainable=True)
        self.WK = tf.Variable(self.__kernel(), name="WK", dtype=tf.float32, trainable=True)
        self.WV = tf.Variable(self.__kernel(), name="WV", dtype=tf.float32, trainable=True)

    def __kernel(self):
        return tf.random.truncated_normal([self.dims, self.key_dims], mean=-0.01, stddev=0.01)

    def __call__(self, queries, keys, values, keep_prob=1.0, mask_future=False):
        drop_rate = 1 - keep_prob

        queries = tf.nn.dropout(queries, rate=drop_rate)
        keys = tf.nn.dropout(keys, rate=drop_rate)
        values = tf.nn.dropout(values, rate=drop_rate)

        head_queries = tf.tensordot(queries, self.WQ, axes=[-1, 0])
        head_keys = tf.tensordot(keys, self.WK, axes=[-1, 0])
        head_values = tf.tensordot(values, self.WV, axes=[-1, 0])

        head = scalar_dot_product_attention(
            query=head_queries,
            key=head_keys,
            value=head_values,
            mask_future=mask_future
        )
        return head


class MultiHeadAttention(object):

    def __init__(self, emb_dims, layers=8):
        self.layer_heads = []
        for i in range(layers):
            with tf.name_scope(f"head-{i}"):
                self.layer_heads.append(SingleHeadAttention(emb_dims=emb_dims, layers=layers))

        self.dense = tf.keras.layers.Dense(units=emb_dims, use_bias=False)

    def __call__(self, queries, keys, values, keep_prob=1.0, mask_future=False):
        heads = [
            layer(queries=queries, keys=keys, values=values, keep_prob=keep_prob, mask_future=mask_future)
            for layer in self.layer_heads
        ]
        total_head = tf.concat(heads, axis=-1)
        return self.dense(total_head)