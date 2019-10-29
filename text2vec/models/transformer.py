from .components.attention import BahdanauAttention
from .components.attention import MultiHeadAttention
from .components.feed_forward import PositionWiseFFN
from .components import utils

import tensorflow as tf
import numpy as np

from functools import partial


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(TransformerEncoder, self).__init__()
        self.__stacks = n_stacks
        self.max_sequence_length = max_sequence_len
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_dropout = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")

        self.positional_encode = utils.positional_encode(emb_dims=dims, max_sequence_length=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.attention = BahdanauAttention(size=dims)

    def __call__(self, inputs, training=False, **kwargs):
        x, mask = inputs
        assert isinstance(x, tf.Tensor)
        assert isinstance(mask, tf.Tensor)

        x = self.dropout(x + (self.positional_encode * mask), training=training)
        for mha, ffn in zip(self.MHA, self.FFN):
            assert isinstance(mha, MultiHeadAttention)
            assert isinstance(ffn, PositionWiseFFN)

            x = self.h_dropout(mha([x] * 3, training=training), training=training) + x
            x = utils.layer_norm_compute(x)
            x = self.h_dropout(ffn(x), training=training) + x
            x = utils.layer_norm_compute(x)

        context = self.attention((x, None))
        if training:
            return x, context
        return context


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_labels, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(TransformerDecoder, self).__init__()
        self.__stacks = n_stacks
        self.max_sequence_length = max_sequence_len
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_dropout = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")

        self.positional_encode = utils.positional_encode(emb_dims=dims, max_sequence_length=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.bias = tf.Variable(tf.zeros([num_labels]), name='bias', dtype=tf.float32, trainable=True)

    def __call__(self, inputs, training=False, **kwargs):
        x_enc, enc_mask, x_dec, dec_mask, context, attention, embeddings = inputs

        assert isinstance(x_enc, tf.Tensor) and isinstance(enc_mask, tf.Tensor)
        assert isinstance(x_dec, tf.Tensor) and isinstance(dec_mask, tf.Tensor)
        assert isinstance(context, tf.Tensor) and isinstance(attention, BahdanauAttention)
        assert isinstance(embeddings, tf.Variable)

        x_dec = self.dropout(x_dec + (self.positional_encode * dec_mask), training=training)
        for mha, ffn in zip(self.MHA, self.FFN):
            assert isinstance(mha, MultiHeadAttention)
            assert isinstance(ffn, PositionWiseFFN)

            x_dec = self.h_dropout(mha([x_dec] * 3, mask_future=True, training=training), training=training) + x_dec
            x_dec = utils.layer_norm_compute(x_dec)

            cross_context = attention((x_enc * enc_mask, x_dec * dec_mask))
            x_dec = self.h_dropout(utils.tensor_projection(x_dec, p_vector=cross_context), training=training) + x_dec

            x_dec = utils.layer_norm_compute(x_dec)
            x_dec = self.h_dropout(ffn(x_dec), training=training) + x_dec
            x_dec = utils.layer_norm_compute(x_dec)
            x_dec = self.h_dropout(utils.tensor_projection(x_dec, p_vector=context), training=training) + x_dec
        return tf.tensordot(x_dec, embeddings, axes=[2, 1]) + self.bias


class Transformer(object):

    def __init__(self, max_sequence_len, token_hash, layers=8, n_stacks=1, embedding_size=50):
        assert isinstance(token_hash, dict)

        self.enc_tokens = tf.compat.v1.placeholder(shape=[None], dtype=tf.string, name='encoder-token-input')
        input_tokens = tf.string_split(self.enc_tokens, sep=' ')
        input_tokens = tf.RaggedTensor.from_sparse(input_tokens)

        self.keep_prob = tf.compat.v1.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))
        self._input_keep_prob, self._hidden_keep_prob, self._dense_keep_prob = tf.unstack(self.keep_prob)
        self._max_seq_len = tf.constant(max_sequence_len, shape=(), dtype=tf.int32)

        self.__use_gpu = tf.test.is_gpu_available()
        self._dims = embedding_size
        self._num_labels = len(token_hash) + 1
        self._layers = layers
        self.__count = 1  # keep track of re-used components for naming purposes
        self.__convolution_count = 1  # keep track of re-used convolution components

        with tf.name_scope('initialize'):
            self.table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(list(token_hash.keys()), list(token_hash.values())),
                default_value=max(token_hash.values()) + 1
            )

            self.embeddings = tf.Variable(
                tf.random.uniform([self._num_labels, embedding_size], -1.0, 1.0),
                name='embeddings',
                dtype=tf.float32,
                trainable=True
            )
            positional_encoder = self.__positional_encoding(max_sequence_len)
            attention = BahdanauAttention(size=embedding_size)
            h_dropout = partial(tf.nn.dropout, rate=1 - self._hidden_keep_prob)

        # Input pipeline
        with tf.name_scope('input'):
            x, enc_mask, _ = self.form_tensors(tokens=input_tokens)
            x = x + positional_encoder * enc_mask
            encoded = tf.nn.dropout(x, rate=1 - self._input_keep_prob)

        # encoder pipeline
        with tf.name_scope('encoder'):
            for _ in range(n_stacks):
                encoded = h_dropout(self.__multi_head_attention(encoded, encoded, encoded)) + encoded
                encoded = utils.layer_norm_compute(encoded)
                encoded = h_dropout(self.__position_wise_feed_forward(encoded)) + encoded
                encoded = utils.layer_norm_compute(encoded)
            self.__context = attention(encoded * enc_mask)

        # Output pipeline
        with tf.name_scope('output'):
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
        with tf.name_scope('decoder'):
            for _ in range(n_stacks):
                decoded = h_dropout(self.__multi_head_attention(decoded, decoded, decoded, mask_future=True)) + decoded
                decoded = utils.layer_norm_compute(decoded)

                cross_context = attention((encoded * enc_mask, decoded * dec_mask))
                decoded = h_dropout(self.__projection(decoded, p_vector=cross_context)) + decoded

                decoded = utils.layer_norm_compute(decoded)
                decoded = h_dropout(self.__position_wise_feed_forward(decoded)) + decoded
                decoded = utils.layer_norm_compute(decoded)
                decoded = h_dropout(self.__projection(decoded)) + decoded

        with tf.name_scope('dense'):
            bias = tf.Variable(
                tf.zeros([self._num_labels]),
                name='bias',
                dtype=tf.float32,
                trainable=True
            )
            x_out = tf.tensordot(decoded, self.embeddings, axes=[2, 1]) + bias  # share embedding weights
            target = target.to_tensor(default_value=0)
            x_out = x_out[:, :dec_time_steps]

        self.loss = self.__cost(target_sequences=target, sequence_logits=x_out, smoothing=False)

        with tf.name_scope('optimizer'):
            self._lr = tf.Variable(0.0, trainable=False)
            self._clip_norm = tf.Variable(0.0, trainable=False)
            t_vars = tf.compat.v1.trainable_variables()
            grads = tf.gradients(self.loss, t_vars)
            opt = tf.compat.v1.train.AdamOptimizer(self._lr)

            # compute the gradient norm - only for logging purposes - remove if greatly affecting performance
            self.gradient_norm = tf.sqrt(sum([tf.norm(t) ** 2 for t in grads]), name="gradient_norm")
            self.train = opt.apply_gradients(zip(grads, t_vars))

            # histograms
            for var in t_vars:
                tf.compat.v1.summary.histogram(var.op.name, var)
            self.merged = tf.compat.v1.summary.merge_all()

            self._new_lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.compat.v1.assign(self._lr, self._new_lr)
            self._new_clip_norm = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_clip_norm")
            self._clip_norm_update = tf.compat.v1.assign(self._clip_norm, self._new_clip_norm)

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

        with tf.name_scope('positional-encoder'):
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
        with tf.name_scope(f'multi-head-attention-{self.__count}'):
            dims = self._dims
            key_dim = dims // self._layers
            heads = []

            # kernel_init = tf.truncated_normal_initializer(-0.01, 0.01)
            def kernel_init(shape): return tf.random.truncated_normal(shape, mean=-0.01, stddev=0.01)

            queries = tf.nn.dropout(queries, rate=1 - self._hidden_keep_prob)
            keys = tf.nn.dropout(keys, rate=1 - self._hidden_keep_prob)
            values = tf.nn.dropout(values, rate=1 - self._hidden_keep_prob)

            for i in range(self._layers):
                with tf.name_scope(f"head-{i}"):
                    # w_q = tf.get_variable("w-query", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    # w_k = tf.get_variable("w-key", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)
                    # w_v = tf.get_variable("w-value", shape=[dims, key_dim], dtype=tf.float32, initializer=kernel_init)

                    w_q = tf.Variable(kernel_init([dims, key_dim]), dtype=tf.float32, name="w-query", trainable=True)
                    w_k = tf.Variable(kernel_init([dims, key_dim]), dtype=tf.float32, name="w-key", trainable=True)
                    w_v = tf.Variable(kernel_init([dims, key_dim]), dtype=tf.float32, name="w-value", trainable=True)

                    head_queries = tf.tensordot(queries, w_q, axes=[-1, 0])
                    head_keys = tf.tensordot(keys, w_k, axes=[-1, 0])
                    head_values = tf.tensordot(values, w_v, axes=[-1, 0])

                    head = utils.scalar_dot_product_attention(
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
        with tf.name_scope(f'position-wise-FFN-{self.__convolution_count}'):
            dims = self._dims
            hidden_dim_size = 4 * dims

            # conv_filter_1 = tf.get_variable('conv-filter-inner', shape=[1, dims, hidden_dim_size], dtype=tf.float32)
            # conv_filter_2 = tf.get_variable('conv-filter-outer', shape=[1, hidden_dim_size, dims], dtype=tf.float32)
            conv_filter_1 = tf.Variable(
                tf.zeros([1, dims, hidden_dim_size]),
                dtype=tf.float32,
                name='conv-filter-inner',
                trainable=True
            )
            conv_filter_2 = tf.Variable(
                tf.zeros([1, hidden_dim_size, dims]),
                dtype=tf.float32,
                name='conv-filter-outer',
                trainable=True
            )

            inner_conv = tf.nn.conv1d(x, filters=conv_filter_1, stride=1, padding='SAME')
            inner_conv = tf.nn.relu(inner_conv)
            outer_conv = tf.nn.conv1d(inner_conv, filters=conv_filter_2, stride=1, padding='SAME')
            self.__convolution_count += 1
            return outer_conv

    def __projection(self, x, p_vector=None):
        with tf.name_scope('projection'):
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
        with tf.name_scope('cost'):
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
