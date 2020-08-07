import tensorflow as tf


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


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, size):
        super(BahdanauAttention, self).__init__(name="BahdanauAttention")
        self.W = tf.Variable(
            tf.random.truncated_normal([size, size], mean=-0.01, stddev=0.01),
            name='weight',
            dtype=tf.float32,
            trainable=True
        )
        self.B = tf.Variable(tf.zeros(shape=[size]), name="B", dtype=tf.float32, trainable=True)
        self.U = tf.Variable(tf.zeros(shape=[size]), name="U", dtype=tf.float32, trainable=True)

    def __call__(self, encoded, decoded=None):
        with tf.name_scope("BahdanauAttention"):
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


class SingleHeadAttention(tf.keras.layers.Layer):

    def __init__(self, emb_dims, layers=8, keep_prob=1.0):
        super(SingleHeadAttention, self).__init__(name="SingleHeadAttention")
        assert isinstance(layers, int) and layers > 0

        dims = emb_dims
        key_dims = emb_dims // layers
        kernel = tf.random.truncated_normal([dims, key_dims], mean=-0.01, stddev=0.01)

        self.WQ = tf.Variable(kernel, name="WQ", dtype=tf.float32, trainable=True)
        self.WK = tf.Variable(kernel, name="WK", dtype=tf.float32, trainable=True)
        self.WV = tf.Variable(kernel, name="WV", dtype=tf.float32, trainable=True)
        self.dropout = tf.keras.layers.Dropout(1 - keep_prob)
        self.dot_attention = ScalarDotAttention()

    def __call__(self, inputs, mask_future=False, training=False):
        with tf.name_scope("SingleHeadAttention"):
            queries, keys, values = inputs

            queries = self.dropout(queries, training=training)
            keys = self.dropout(keys, training=training)
            values = self.dropout(values, training=training)

            head_queries = tf.tensordot(queries, self.WQ, axes=[-1, 0])
            head_keys = tf.tensordot(keys, self.WK, axes=[-1, 0])
            head_values = tf.tensordot(values, self.WV, axes=[-1, 0])
            return self.dot_attention(query=head_queries, key=head_keys, value=head_values, mask_future=mask_future)


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, emb_dims, layers=8, keep_prob=1.0):
        super(MultiHeadAttention, self).__init__(name="MultiHeadAttention")
        self.layer_heads = []
        for i in range(layers):
            with tf.name_scope(f"head-{i}"):
                self.layer_heads.append(SingleHeadAttention(emb_dims=emb_dims, layers=layers, keep_prob=keep_prob))

        self.dense = tf.keras.layers.Dense(units=emb_dims, use_bias=False)

    def __call__(self, inputs, mask_future=False, training=False):
        with tf.name_scope("MultiHeadAttention"):
            heads = [layer(inputs, mask_future=mask_future, training=training) for layer in self.layer_heads]
            total_head = tf.concat(heads, axis=-1)
            return self.dense(total_head)
