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
        self._num_labels = vocab_size
        self._layers = layers
        self._input_keep_prob, self._hidden_keep_prob, self._dense_keep_prob = tf.unstack(keep_prob)

        embeddings = tf.Variable(
            tf.random_uniform([vocab_size, self._dims], -1.0, 1.0),
            name='word-embeddings',
            dtype=tf.float32,
            trainable=True
        )

        # Input pipeline
        with tf.variable_scope('input'):
            self._seq_lengths = tf.count_nonzero(input_x, axis=1, name='sequence-lengths')
            x = tf.nn.embedding_lookup(embeddings, input_x)
            encoding = self.__positional_encoding(x)
            x = x + encoding
            x = tf.nn.dropout(x, keep_prob=self._input_keep_prob)

        with tf.variable_scope('encoder'):
            x_encoded = self.__multi_head_attention(queries=x, keys=x, values=x) + x
            x_encoded = self.layer_norm_compute(x_encoded)
            x_encoded = self.__point_wise_feed_forward(x_encoded) + x_encoded
            x_encoded = self.layer_norm_compute(x_encoded)

        # Output pipeline
        with tf.variable_scope('output'):
            if is_training:
                decode_x = tf.concat([input_x[:, 1:], tf.zeros_like(input_x[:, :1])], axis=1)
            else:
                decode_x = tf.zeros_like(input_x, dtype=tf.float32)
            x = tf.nn.embedding_lookup(embeddings, decode_x)
            encoding = self.__positional_encoding(x)
            x = x + encoding
            x = tf.nn.dropout(x, keep_prob=self._input_keep_prob)

        with tf.variable_scope('decoder'):
            # todo: in the first multi-head-attention mask out future sequence values so learning only depends on past
            x_decoded = self.__masked_multi_head_attention(values=x, keys=x, queries=x) + x
            x_decoded = self.layer_norm_compute(x_decoded)
            x_decoded = self.__multi_head_attention(values=x_encoded, keys=x_encoded, queries=x_decoded) + x_decoded
            x_decoded = self.layer_norm_compute(x_decoded)
            x_decoded = self.__point_wise_feed_forward(x_decoded) + x_decoded
            x_decoded = self.layer_norm_compute(x_decoded)

        self.__context = self.__bahdanau_attention(encoded=x_encoded, decoded=x_decoded)

        if is_training:
            with tf.variable_scope('sequence-reconstructor'):
                x_out = self.__projection(x_decoded)
                x_out = tf.layers.dense(inputs=x_out, units=vocab_size, name="output-dense")

            with tf.variable_scope('cost'):
                self.loss = self.__cost(target_sequences=input_x, sequence_logits=x_out)

            with tf.variable_scope('optimizer'):
                self._lr = tf.Variable(0.0, trainable=False)
                self._clip_norm = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()

                # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
                grads = tf.gradients(self.loss, t_vars)
                opt = tf.train.AdamOptimizer(self._lr, beta1=0.9, beta2=0.98, epsilon=1e-9)

                # compute the gradient norm - only for logging purposes - remove if greatly affecting performance
                self.gradient_norm = tf.sqrt(sum([tf.norm(t) ** 2 for t in grads]), name="gradient_norm")

                self.train = opt.apply_gradients(
                    zip(grads, t_vars),
                    global_step=tf.train.get_or_create_global_step()
                )

                self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
                self._lr_update = tf.assign(self._lr, self._new_lr)

                self._new_clip_norm = tf.placeholder(tf.float32, shape=[], name="new_clip_norm")
                self._clip_norm_update = tf.assign(self._clip_norm, self._new_clip_norm)

    def assign_lr(self, session, lr_value):
        """
        Updates the learning rate
        :param session: (TensorFlow Session)
        :param lr_value: (float)
        """

        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_clip_norm(self, session, norm_value):
        """
        Updates the gradient normalization factor
        :param session: (TensorFlow Session)
        :param norm_value: (float)
        """

        session.run(self._clip_norm_update, feed_dict={self._new_clip_norm: norm_value})

    def __positional_encoding(self, x):
        """
        Positional encoding according to https://arxiv.org/pdf/1706.03762.pdf
        :param x: batch tensor (tf.Tensor)
        :return: single batch encoder (tf.Tensor)
        """

        with tf.variable_scope('positional-encoder'):
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
        with tf.variable_scope('scalar-dot-attention'):
            numerator = tf.einsum('ijk,ilk->ijl', query, key)
            denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

            x_ = tf.nn.softmax(numerator / denominator)
            return tf.einsum('ijk,ikl->ijl', x_, value)

    def __multi_head_attention(self, queries, keys, values):
        with tf.variable_scope('multi-head-attention'):
            key_dim = self._dims // self._layers
            heads = []
            queries = tf.nn.dropout(queries, self._hidden_keep_prob)
            keys = tf.nn.dropout(keys, self._hidden_keep_prob)
            values = tf.nn.dropout(values, self._hidden_keep_prob)

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
                        query=tf.einsum('ijk,kl->ijl', queries, w_q),
                        key=tf.einsum('ijk,kl->ijl', keys, w_k),
                        value=tf.einsum('ijk,kl->ijl', values, w_v)
                    )
                    heads.append(head)

            total_head = tf.concat(heads, axis=-1)
            return tf.einsum('ijk,kl->ijl', total_head, w)

    def __masked_multi_head_attention(self, queries, keys, values):
        with tf.variable_scope('masked-multi-head-attention'):
            key_dim = self._dims // self._layers
            heads = []
            queries = tf.nn.dropout(queries, self._hidden_keep_prob)
            keys = tf.nn.dropout(keys, self._hidden_keep_prob)
            values = tf.nn.dropout(values, self._hidden_keep_prob)

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
                        query=tf.einsum('ijk,kl->ijl', queries, w_q),
                        key=tf.einsum('ijk,kl->ijl', keys, w_k),
                        value=tf.einsum('ijk,kl->ijl', values, w_v)
                    )
                    heads.append(head)

            total_head = tf.concat(heads, axis=-1)
            return tf.einsum('ijk,kl->ijl', total_head, w)

    def __point_wise_feed_forward(self, x):
        with tf.variable_scope('FFN'):
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
            output = tf.nn.relu(tf.einsum('ijk,lk->ijk', x, w_1) + b_1)
            output = tf.einsum('ijk,lk->ijk', output, w_2) + b_2
            return output

    @staticmethod
    def layer_norm_compute(x, epsilon=1e-6, scale=1.0, bias=0):
        with tf.variable_scope('layer-norm'):
            mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
            return norm_x * scale + bias

    def __bahdanau_attention(self, encoded, decoded):
        with tf.variable_scope('bahdanau-attention'):
            weight = tf.get_variable(
                "weight",
                shape=[self._time_steps, self._dims],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            b_omega = tf.get_variable("b_omega", shape=[self._dims], initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u_omega", shape=[self._dims], initializer=tf.zeros_initializer())

            v = tf.einsum('ilk,ld,ijd->ijk', encoded, weight, decoded) + b_omega
            vu = tf.einsum("ijl,l->ij", v, u_omega, name="Bahdanau_score")
            alphas = tf.nn.softmax(vu, name="attention_weights")

            output = tf.reduce_sum(encoded * tf.expand_dims(alphas, -1), 1, name="context_vector")
            return output

    def __projection(self, x):
        with tf.variable_scope('projection'):
            inner_product = tf.einsum("ijk,ik->ij", x, self.__context)
            context_norm_squared = tf.norm(self.__context, axis=1) ** 2

            # to make this work on the GPU we can't do broadcasting
            # so we need to expand context norm to a 2D tensor,
            # then tile that out to have the same number of columns as inner product
            context_norm_squared = tf.tile(tf.expand_dims(context_norm_squared, -1), [1, self._time_steps])

            alpha = tf.divide(inner_product, context_norm_squared)
            projection = tf.einsum("ij,ik->ijk", alpha, self.__context, name="projection_op")
            return projection

    def __cost(self, target_sequences, sequence_logits):
        epsilon = tf.constant(1e-8, dtype=tf.float32, shape=[1])
        length = tf.cast(self._seq_lengths, dtype=tf.float32)
        length = tf.add(length, epsilon)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=sequence_logits,
            labels=target_sequences
        )

        # smoothing = 0.1
        # targets = tf.one_hot(target_sequences, depth=self._num_labels, on_value=1.0, off_value=0.0, axis=-1)
        # loss = tf.losses.softmax_cross_entropy(
        #     logits=sequence_logits,
        #     onehot_labels=targets,
        #     label_smoothing=smoothing,
        #     reduction=tf.losses.Reduction.NONE
        # )

        loss = tf.reduce_sum(loss, axis=-1) / length
        loss = tf.reduce_mean(loss)
        return loss

    @property
    def embedding(self):
        return self.__context
