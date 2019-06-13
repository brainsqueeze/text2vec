import tensorflow as tf
import numpy as np

from . import model_utils as tf_utils
from functools import partial


class Attention(object):

    def __init__(self, size):
        self.weight = tf.get_variable(
            "weight",
            dtype=tf.float32,
            shape=[size, size],
            initializer=tf.truncated_normal_initializer(-0.01, 0.01)
        )
        self.b_omega = tf.get_variable("b_omega", shape=[size], initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.u_omega = tf.get_variable("u_omega", shape=[size], initializer=tf.zeros_initializer(), dtype=tf.float32)

    def __call__(self, encoded, decoded=None):
        with tf.variable_scope('bahdanau-attention'):
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


class Transformer(object):

    def __init__(self, max_sequence_len, token_hash, layers=8, n_stacks=1, embedding_size=50):
        assert isinstance(token_hash, dict)

        self.enc_tokens = tf.placeholder(shape=[None], dtype=tf.string, name='encoder-token-input')
        input_tokens = tf.string_split(self.enc_tokens, delimiter=' ')
        input_tokens = tf.RaggedTensor.from_sparse(input_tokens)

        self.keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))
        self._input_keep_prob, self._hidden_keep_prob, self._dense_keep_prob = tf.unstack(self.keep_prob)
        self._max_seq_len = tf.constant(max_sequence_len, shape=(), dtype=tf.int32)

        self.__use_gpu = tf.test.is_gpu_available()
        self._dims = embedding_size
        self._num_labels = len(token_hash) + 1
        self._layers = layers
        self.__count = 1  # keep track of re-used components for naming purposes
        self.__convolution_count = 1  # keep track of re-used convolution components

        with tf.variable_scope('initialize'):
            self.table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(list(token_hash.keys()), list(token_hash.values())),
                default_value=max(token_hash.values()) + 1
            )

            self.embeddings = tf.Variable(
                tf.random_uniform([self._num_labels, embedding_size], -1.0, 1.0),
                name='embeddings',
                dtype=tf.float32,
                trainable=True
            )
            positional_encoder = self.__positional_encoding(max_sequence_len)
            attention = Attention(size=embedding_size)
            h_dropout = partial(tf.nn.dropout, rate=1 - self._hidden_keep_prob)

        # Input pipeline
        with tf.variable_scope('input'):
            x, enc_mask, _ = self.form_tensors(tokens=input_tokens)
            x = x + positional_encoder * enc_mask
            encoded = tf.nn.dropout(x, rate=1 - self._input_keep_prob)

        # encoder pipeline
        with tf.variable_scope('encoder'):
            for _ in range(n_stacks):
                encoded = h_dropout(self.__multi_head_attention(encoded, encoded, encoded)) + encoded
                encoded = tf_utils.layer_norm_compute(encoded)
                encoded = h_dropout(self.__position_wise_feed_forward(encoded)) + encoded
                encoded = tf_utils.layer_norm_compute(encoded)
            self.__context = attention(encoded * enc_mask)

        # Output pipeline
        with tf.variable_scope('output'):
            # make targets
            batch_size = input_tokens.nrows()
            eos = tf.fill([batch_size], value='</s>')  # post-pend EOS tag
            eos = tf.expand_dims(eos, axis=-1)
            target = tf.concat([input_tokens, eos], axis=-1)
            target = tf.ragged.map_flat_values(self.table.lookup, target)
            target = target[:, :self._max_seq_len]

            # make decoding sequences
            bos = tf.fill([batch_size], value='<s>')
            bos = tf.expand_dims(bos, axis=-1)
            dec_tokens = tf.concat([bos, input_tokens], axis=-1)  # prepend BOS tag
            x, dec_mask, dec_time_steps = self.form_tensors(tokens=dec_tokens)
            x = x + positional_encoder * dec_mask
            decoded = tf.nn.dropout(x, rate=1 - self._input_keep_prob)

        # decoder pipeline
        with tf.variable_scope('decoder'):
            for _ in range(n_stacks):
                decoded = h_dropout(self.__multi_head_attention(decoded, decoded, decoded, mask_future=True)) + decoded
                decoded = tf_utils.layer_norm_compute(decoded)

                cross_context = attention(encoded=encoded * enc_mask, decoded=decoded * dec_mask)
                decoded = h_dropout(self.__projection(decoded, p_vector=cross_context)) + decoded

                decoded = tf_utils.layer_norm_compute(decoded)
                decoded = h_dropout(self.__position_wise_feed_forward(decoded)) + decoded
                decoded = tf_utils.layer_norm_compute(decoded)
                decoded = h_dropout(self.__projection(decoded)) + decoded

        with tf.variable_scope('dense'):
            bias = tf.get_variable(
                "bias",
                dtype=tf.float32,
                shape=[self._num_labels],
                initializer=tf.zeros_initializer()
            )
            x_out = tf.tensordot(decoded, self.embeddings, axes=[2, 1]) + bias  # share embedding weights
            target = target.to_tensor(default_value=0)
            x_out = x_out[:, :dec_time_steps]

        self.loss = self.__cost(target_sequences=target, sequence_logits=x_out, smoothing=False)

        with tf.variable_scope('optimizer'):
            self._lr = tf.Variable(0.0, trainable=False)
            self._clip_norm = tf.Variable(0.0, trainable=False)
            t_vars = tf.trainable_variables()
            grads = tf.gradients(self.loss, t_vars)
            opt = tf.train.AdamOptimizer(self._lr)

            # compute the gradient norm - only for logging purposes - remove if greatly affecting performance
            self.gradient_norm = tf.sqrt(sum([tf.norm(t) ** 2 for t in grads]), name="gradient_norm")
            self.train = opt.apply_gradients(zip(grads, t_vars), global_step=tf.train.get_or_create_global_step())

            # histograms
            for var in t_vars:
                tf.summary.histogram(var.op.name, var)
            self.merged = tf.summary.merge_all()

            self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
            self._new_clip_norm = tf.placeholder(tf.float32, shape=[], name="new_clip_norm")
            self._clip_norm_update = tf.assign(self._clip_norm, self._new_clip_norm)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_clip_norm(self, session, norm_value):
        session.run(self._clip_norm_update, feed_dict={self._new_clip_norm: norm_value})

    def form_tensors(self, tokens):
        epsilon = 1e-8

        hashed = tf.ragged.map_flat_values(self.table.lookup, tokens)
        hashed = hashed[:, :self._max_seq_len]
        batch_size = hashed.nrows()
        seq_lengths = hashed.row_lengths()
        time_steps = tf.cast(tf.reduce_max(seq_lengths), dtype=tf.int32)
        padding = tf.zeros(shape=(batch_size, self._max_seq_len - time_steps, self._dims), dtype=tf.float32)

        x = tf.ragged.map_flat_values(tf.nn.embedding_lookup, self.embeddings, hashed)
        x = x.to_tensor()
        # pad to full max sequence length, otherwise we get numerical inconsistencies with differing batch sizes
        x = tf.concat([x, padding], axis=1)

        # time-step masking
        dec_mask = tf.sequence_mask(lengths=seq_lengths, maxlen=self._max_seq_len)
        dec_mask = tf.cast(dec_mask, dtype=tf.float32)
        dec_mask = tf.tile(tf.expand_dims(dec_mask, axis=-1), multiples=[1, 1, self._dims]) + epsilon
        return x, dec_mask, time_steps

    def __positional_encoding(self, max_seq_len):
        """
        Positional encoding according to https://arxiv.org/pdf/1706.03762.pdf
        :return: single batch encoder (tf.Tensor)
        """

        with tf.variable_scope('positional-encoder'):
            positions = np.arange(max_seq_len).astype(np.float32)
            column_range = np.arange(self._dims).astype(np.float32)
            factor = np.power(1e5 ** (2 / self._dims), column_range)

            even = np.sin(positions / factor[::2, None]).T
            odd = np.cos(positions / factor[1::2, None]).T

            encoder = np.zeros(shape=(max_seq_len, self._dims), dtype=np.float32)
            encoder[:, ::2] = even
            encoder[:, 1::2] = odd
            encoder = tf.convert_to_tensor(encoder, dtype=tf.float32)
            return encoder

    def __multi_head_attention(self, queries, keys, values, mask_future=False):
        with tf.variable_scope(f'multi-head-attention-{self.__count}'):
            dims = self._dims
            key_dim = dims // self._layers
            heads = []

            kernel_init = tf.truncated_normal_initializer(-0.01, 0.01)

            queries = tf.nn.dropout(queries, rate=1 - self._hidden_keep_prob)
            keys = tf.nn.dropout(keys, rate=1 - self._hidden_keep_prob)
            values = tf.nn.dropout(values, rate=1 - self._hidden_keep_prob)

            for i in range(self._layers):
                with tf.variable_scope(f"head-{i}"):
                    w_q = tf.get_variable("w-query", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    w_k = tf.get_variable("w-key", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    w_v = tf.get_variable("w-value", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)

                    head_queries = tf.tensordot(queries, w_q, axes=[-1, 0])
                    head_keys = tf.tensordot(keys, w_k, axes=[-1, 0])
                    head_values = tf.tensordot(values, w_v, axes=[-1, 0])

                    head = tf_utils.scalar_dot_product_attention(
                        query=head_queries,
                        key=head_keys,
                        value=head_values,
                        mask_future=mask_future
                    )
                    heads.append(head)

            total_head = tf.concat(heads, axis=-1)
            self.__count += 1
            output = tf.keras.layers.Dense(units=dims, use_bias=False).apply(inputs=total_head)
            return output

    def __position_wise_feed_forward(self, x):
        with tf.variable_scope(f'position-wise-FFN-{self.__convolution_count}'):
            dims = self._dims
            hidden_dim_size = 4 * dims

            conv_filter_1 = tf.get_variable('conv-filter-inner', shape=[1, dims, hidden_dim_size], dtype=tf.float32)
            conv_filter_2 = tf.get_variable('conv-filter-outer', shape=[1, hidden_dim_size, dims], dtype=tf.float32)

            inner_conv = tf.nn.conv1d(x, filters=conv_filter_1, stride=1, padding='SAME')
            inner_conv = tf.nn.relu(inner_conv)
            outer_conv = tf.nn.conv1d(inner_conv, filters=conv_filter_2, stride=1, padding='SAME')
            self.__convolution_count += 1
            return outer_conv

    def __projection(self, x, p_vector=None):
        with tf.variable_scope('projection'):
            assert isinstance(x, tf.Tensor)

            context = p_vector if p_vector is not None else self.__context
            inner_product = tf.einsum("ijk,ik->ij", x, context)
            context_norm_squared = tf.norm(context, axis=1) ** 2
            time_steps = tf.shape(x)[1]

            # to make this work on the GPU we can't do broadcasting
            # so we need to expand context norm to a 2D tensor,
            # then tile that out to have the same number of columns as inner product
            context_norm_squared = tf.tile(tf.expand_dims(context_norm_squared, -1), [1, time_steps])

            alpha = tf.divide(inner_product, context_norm_squared)
            projection = tf.einsum("ij,ik->ijk", alpha, context, name="projection_op")
            return projection

    def __cost(self, target_sequences, sequence_logits, smoothing=False):
        with tf.variable_scope('cost'):
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

            loss = tf.reduce_mean(loss)
            return loss

    @property
    def embedding(self):
        return self.__context

    @property
    def lr(self):
        return self._lr
