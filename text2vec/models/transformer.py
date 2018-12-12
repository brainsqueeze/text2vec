import tensorflow as tf


class Tensor2Tensor(object):
    """
    This model is based off of the tensor-to-tensor transformer
    model described in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, input_x, vocab_size, embedding_size, keep_prob, layers=8, is_training=False):
        self.__use_gpu = tf.test.is_gpu_available()

        self._batch_size, self._time_steps = input_x.get_shape().as_list()
        self._dims = embedding_size
        self._layers = layers
        self._input_keep_prob, _, self._dense_keep_prob = tf.unstack(keep_prob)

        with tf.variable_scope('embedding'):
            self._seq_lengths = tf.count_nonzero(input_x, axis=1, name='sequence-lengths')
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, self._dims], -1.0, 1.0),
                name='word-embeddings',
                dtype=tf.float32,
                trainable=True
            )
            x = tf.nn.embedding_lookup(embeddings, input_x)
            x = tf.nn.dropout(x, keep_prob=self._input_keep_prob)

        with tf.variable_scope('positional-encoder'):
            encoding = self.__positional_encoding(x)
            x = x + encoding

        with tf.variable_scope('encoder'):
            x_encoded = self.__multi_head_attention(x) + x
            x_encoded = self.layer_norm_compute(x_encoded)
            x_encoded = self.__point_wise_feed_forward(x_encoded) + x_encoded
            x_encoded = self.layer_norm_compute(x_encoded)

        self.__embedded = self.__attention(x_encoded)

        if is_training:
            with tf.variable_scope('sequence-reconstructor'):
                x_out = self.__projection(x_encoded)
                inner = tf.einsum('ijk,lk->ijl', x_out, embeddings)
                inner /= tf.linalg.norm(inner, axis=-1, keepdims=True)

                max_args = tf.argmax(inner, axis=-1, output_type=tf.int32)
                loss = self.__cost(expected_sequence=input_x, reconstructed_sequence=max_args)
                print(loss)

    def __positional_encoding(self, x):
        """
        Positional encoding according to https://arxiv.org/pdf/1706.03762.pdf
        :param x: batch tensor (tf.Tensor)
        :return: single batch encoder (tf.Tensor)
        """

        positions = tf.range(self._time_steps, dtype=tf.float32)
        column_range = tf.range(self._dims, dtype=tf.float32)
        factor = tf.pow(tf.constant(1e5 ** (2 / self._dims), dtype=tf.float32), column_range)

        # encoding
        even = tf.transpose(tf.sin(positions / factor[::2, tf.newaxis]))
        odd = tf.transpose(tf.cos(positions / factor[1::2, tf.newaxis]))

        # inter-weave
        encoder = tf.Variable(tf.zeros_like(x[0]), dtype=tf.float32, trainable=False)
        encoder[:, ::2].assign(even)
        encoder[:, 1::2].assign(odd)
        return encoder

    @staticmethod
    def scalar_dot_product_attention(query, key, value):
        numerator = tf.einsum('ijk,ilk->ijl', query, key)
        denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

        x_ = tf.nn.softmax(numerator / denominator)
        return tf.einsum('ijk,ikl->ijl', x_, value)

    def __multi_head_attention(self, x):
        key_dim = self._dims // self._layers
        heads = []

        w = tf.get_variable(
            "total-head-weight",
            shape=[self._layers * key_dim, self._dims],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(-0.01, 0.01)
        )

        for i in range(self._layers):
            with tf.variable_scope("head-{layer}".format(layer=i)):
                w_q = tf.get_variable(
                    "query-weight",
                    shape=[self._dims, key_dim],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(-0.01, 0.01)
                )

                w_k = tf.get_variable(
                    "key-weight",
                    shape=[self._dims, key_dim],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(-0.01, 0.01)
                )

                w_v = tf.get_variable(
                    "value-weight",
                    shape=[self._dims, key_dim],
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(-0.01, 0.01)
                )

                head = self.scalar_dot_product_attention(
                    query=tf.einsum('ijk,kl->ijl', x, w_q),
                    key=tf.einsum('ijk,kl->ijl', x, w_k),
                    value=tf.einsum('ijk,kl->ijl', x, w_v)
                )
                heads.append(head)

        total_head = tf.concat(heads, axis=-1)
        return tf.einsum('ijk,kl->ijl', total_head, w)

    def __point_wise_feed_forward(self, x):
        w_1 = tf.get_variable(
            "inner-weight",
            shape=[self._dims, self._dims],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(-0.01, 0.01)
        )
        b_1 = tf.get_variable(
            "inner-bias",
            shape=[self._dims],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        w_2 = tf.get_variable(
            "outer-weight",
            shape=[self._dims, self._dims],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(-0.01, 0.01)
        )
        b_2 = tf.get_variable(
            "outer-bias",
            shape=[self._dims],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )
        # output = tf.nn.relu(tf.nn.xw_plus_b(x, weights=w_1, biases=b_1))
        # output = tf.nn.xw_plus_b(output, weights=w_2, biases=b_2)

        output = tf.nn.relu(tf.einsum('ijk,lk->ijk', x, w_1) + b_1)
        output = tf.einsum('ijk,lk->ijk', output, w_2) + b_2
        return output

    @staticmethod
    def layer_norm_compute(x, epsilon=1e-6, scale=1.0, bias=0):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias

    def __attention(self, x):
        with tf.variable_scope('bahdanau-self-attention'):
            hidden_size = self._dims // 2

            in_dim = x.get_shape().as_list()[-1]
            w_omega = tf.get_variable(
                "w_omega",
                shape=[in_dim, hidden_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            b_omega = tf.get_variable("b_omega", shape=[hidden_size], initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u_omega", shape=[hidden_size], initializer=tf.zeros_initializer())

            v = tf.tanh(tf.einsum("ijk,kl->ijl", x, w_omega) + b_omega)
            vu = tf.einsum("ijl,l->ij", v, u_omega, name="Bahdanau_score")
            alphas = tf.nn.softmax(vu, name="attention_weights")

            output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), 1, name="context_vector")
            return output

    def __projection(self, x):
        with tf.variable_scope('projection'):
            inner_product = tf.einsum("ijk,ik->ij", x, self.__embedded)
            context_norm_sqrd = tf.norm(self.__embedded, axis=1) ** 2

            # to make this work on the GPU we can't do broadcasting
            # so we need to expand context norm to a 2D tensor,
            # then tile that out to have the same number of columns as inner product
            context_norm_sqrd = tf.tile(tf.expand_dims(context_norm_sqrd, -1), [1, self._time_steps])

            alpha = tf.divide(inner_product, context_norm_sqrd)
            projection = tf.einsum("ij,ik->ijk", alpha, self.__embedded, name="projection_op")
            return projection

    def __cost(self, expected_sequence, reconstructed_sequence):
        with tf.variable_scope('cost'):
            epsilon = tf.constant(1e-8, dtype=tf.float32, shape=[1])
            length = tf.cast(self._seq_lengths, dtype=tf.float32)
            length = tf.add(length, epsilon)

            expected_sequence = tf.cast(expected_sequence, dtype=tf.float32)
            reconstructed_sequence = tf.cast(reconstructed_sequence, dtype=tf.float32)

            mse = (expected_sequence - reconstructed_sequence) ** 2
            mse = tf.reduce_sum(mse, axis=-1) / length
            mse = tf.reduce_mean(mse)

            return mse
