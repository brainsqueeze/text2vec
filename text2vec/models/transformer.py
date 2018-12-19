import tensorflow as tf


class Tensor2Tensor(object):
    """
    This model is based off of the tensor-to-tensor transformer
    model described in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, max_sequence_len, vocab_size, embedding_size, layers=8, word_weights=None, is_training=False):
        self.seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len], name='input')
        self.keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

        self.__use_gpu = tf.test.is_gpu_available()

        self._batch_size, self._time_steps = self.seq_input.get_shape().as_list()
        self._dims = embedding_size
        self._num_labels = vocab_size
        self._layers = layers
        self._input_keep_prob, self._hidden_keep_prob, self._dense_keep_prob = tf.unstack(self.keep_prob)

        self.__count = 1  # keep track of re-used components for naming purposes

        if is_training and word_weights is not None:
            embeddings = tf.Variable(word_weights, name='embeddings', dtype=tf.float32, trainable=False)
            self._dims = word_weights.shape[1]
        else:
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, self._dims], -1.0, 1.0),
                name='embeddings',
                dtype=tf.float32,
                trainable=True
            )

        # Input pipeline
        with tf.variable_scope('input'):
            self._seq_lengths = tf.count_nonzero(self.seq_input, axis=1, name='sequence-lengths')
            inputs = tf.nn.embedding_lookup(embeddings, self.seq_input)
            encoding = self.__positional_encoding(inputs)
            x = inputs + encoding
            x = tf.nn.dropout(x, keep_prob=self._input_keep_prob)

        with tf.variable_scope('encoder'):
            x_encoded = self.__multi_head_attention(queries=x, keys=x, values=x) + x
            x_encoded = self.layer_norm_compute(x_encoded)
            x_encoded = self.__position_wise_feed_forward(x_encoded) + x_encoded
            x_encoded = self.layer_norm_compute(x_encoded)
            self.__context = self.__bahdanau_attention(encoded=x_encoded)

        # Output pipeline
        with tf.variable_scope('output'):
            decode_x = tf.concat([tf.ones_like(self.seq_input[:, :1]), self.seq_input[:, :-1]], axis=1)  # shift right

            x = tf.nn.embedding_lookup(embeddings, decode_x)
            encoding = self.__positional_encoding(x)
            x = x + encoding
            x = tf.nn.dropout(x, keep_prob=self._input_keep_prob)

        with tf.variable_scope('decoder'):
            x_decoded = self.__multi_head_attention(values=x, keys=x, queries=x, mask_future=True) + x
            x_decoded = self.layer_norm_compute(x_decoded)

            x_decoded = self.__projection(x_decoded) + x_decoded
            x_decoded = self.layer_norm_compute(x_decoded)

            x_decoded = self.__position_wise_feed_forward(x_decoded) + x_decoded
            x_decoded = self.layer_norm_compute(x_decoded)
            x_decoded = self.__projection(x_decoded) + x_decoded

        if is_training:
            x_out = tf.layers.dense(inputs=x_decoded, units=vocab_size, name="dense")
            self.loss = self.__cost(target_sequences=self.seq_input, sequence_logits=x_out)

            with tf.variable_scope('optimizer'):
                self._lr = tf.Variable(0.0, trainable=False)
                self._clip_norm = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()

                # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
                grads = tf.gradients(self.loss, t_vars)
                # opt = tf.train.AdamOptimizer(self._lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
                opt = tf.train.AdamOptimizer(self._lr)

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

            mask = tf.sequence_mask(lengths=self._seq_lengths, maxlen=self._time_steps, name='encoding-mask')
            mask = tf.cast(mask, dtype=tf.float32)
            encoder = encoder * tf.tile(tf.expand_dims(mask, axis=-1), multiples=[1, 1, self._dims])

            return encoder

    @staticmethod
    def scalar_dot_product_attention(query, key, value, mask_future=False):
        with tf.variable_scope('scalar-dot-attention'):
            numerator = tf.einsum('ijk,ilk->ijl', query, key)
            denominator = tf.sqrt(tf.cast(tf.shape(key)[1], dtype=tf.float32))

            if mask_future:
                upper = (1 + 1e9) * tf.linalg.band_part(tf.ones_like(numerator), num_lower=0, num_upper=-1)
                mask = 1 - upper
                numerator *= mask

            x_ = tf.nn.softmax(numerator / denominator)
            return tf.einsum('ijk,ikl->ijl', x_, value)

    def __multi_head_attention(self, queries, keys, values, mask_future=False):
        with tf.variable_scope('multi-head-attention-{iteration}'.format(iteration=self.__count)):
            dims = self._dims
            key_dim = dims // self._layers
            heads = []

            kernel_init = tf.truncated_normal_initializer(-0.01, 0.01)

            queries = tf.nn.dropout(queries, self._hidden_keep_prob)
            keys = tf.nn.dropout(keys, self._hidden_keep_prob)
            values = tf.nn.dropout(values, self._hidden_keep_prob)

            for i in range(self._layers):
                with tf.variable_scope("head-{layer}".format(layer=i)):
                    w_q = tf.get_variable("w-query", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    w_k = tf.get_variable("w-key", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    w_v = tf.get_variable("w-value", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)

                    head_queries = tf.einsum('ijk,kl->ijl', queries, w_q)
                    head_keys = tf.einsum('ijk,kl->ijl', keys, w_k)
                    head_values = tf.einsum('ijk,kl->ijl', values, w_v)

                    head = self.scalar_dot_product_attention(
                        query=head_queries,
                        key=head_keys,
                        value=head_values,
                        mask_future=mask_future
                    )
                    heads.append(head)

            total_head = tf.concat(heads, axis=-1)
            self.__count += 1
            output = tf.layers.dense(inputs=total_head, units=dims, use_bias=False)
            return output

    def __position_wise_feed_forward(self, x):
        with tf.variable_scope('position-wise-FFN'):
            dims = self._dims
            hidden_dim_size = 4 * dims

            conv_filter_1 = tf.get_variable('conv-filter-inner', shape=[1, dims, hidden_dim_size], dtype=tf.float32)
            conv_filter_2 = tf.get_variable('conv-filter-outer', shape=[1, hidden_dim_size, dims], dtype=tf.float32)

            inner_conv = tf.nn.conv1d(x, filters=conv_filter_1, stride=1, padding='SAME')
            inner_conv = tf.nn.relu(inner_conv)
            outer_conv = tf.nn.conv1d(inner_conv, filters=conv_filter_2, stride=1, padding='SAME')
            return outer_conv

    @staticmethod
    def layer_norm_compute(x, epsilon=1e-8, scale=1.0, bias=0):
        with tf.variable_scope('layer-norm'):
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
            norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
            return norm_x * scale + bias

    def __bahdanau_attention(self, encoded, decoded=None):
        with tf.variable_scope('bahdanau-attention'):
            kernel_init = tf.truncated_normal_initializer(-0.01, 0.01)

            if decoded is None:
                attention_size = self._dims
                weight = tf.get_variable("weight", shape=[self._dims, attention_size], initializer=kernel_init)
                b_omega = tf.get_variable("b_omega", shape=[attention_size], initializer=tf.zeros_initializer())
                u_omega = tf.get_variable("u_omega", shape=[attention_size], initializer=tf.zeros_initializer())
                score = tf.tanh(tf.einsum("ijk,kl->ijl", encoded, weight) + b_omega)
            else:
                weight = tf.get_variable("weight", shape=[self._time_steps], initializer=kernel_init)
                u_omega = tf.get_variable("u_omega", shape=[self._dims], initializer=tf.zeros_initializer())
                processed_query = tf.expand_dims(tf.einsum('m,imj->ij', weight, decoded), axis=1)
                score = tf.tanh(processed_query + encoded)

            score = tf.reduce_sum(u_omega * score, axis=-1)
            alphas = tf.nn.softmax(score, name="attention-weights")

            output = tf.reduce_sum(encoded * tf.expand_dims(alphas, -1), 1, name="context-vector")
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

    def __cost(self, target_sequences, sequence_logits, smoothing=False):
        with tf.variable_scope('cost'):
            epsilon = tf.constant(1e-8, dtype=tf.float32, shape=[1])
            length = tf.cast(self._seq_lengths, dtype=tf.float32)
            length = tf.add(length, epsilon)

            if smoothing:
                smoothing = 0.1
                targets = tf.one_hot(target_sequences, depth=self._num_labels, on_value=1.0, off_value=0.0, axis=-1)
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

            loss = tf.reduce_sum(loss, axis=-1) / length
            loss = tf.reduce_mean(loss)
            return loss

    @property
    def embedding(self):
        return self.__context

    @property
    def lr(self):
        return self._lr
