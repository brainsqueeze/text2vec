from .components.attention import BahdanauAttention
import tensorflow as tf

from functools import partial


class Sequential(object):

    def __init__(self, max_sequence_len, token_hash, num_hidden, embedding_size=50):
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
        self._num_hidden = num_hidden

        with tf.name_scope('initializer'):
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
            attention = BahdanauAttention(size=2 * self._num_hidden)
            h_dropout = partial(tf.nn.dropout, rate=1 - self._hidden_keep_prob)
            self.__add_layer = tf.keras.layers.Add()
            self.__dense_layer = tf.keras.layers.Dense(units=self._num_labels, dtype=tf.float32)

        with tf.name_scope('input'):
            x, enc_mask, _ = self.form_tensors(tokens=input_tokens)
            encoded = tf.nn.dropout(x, rate=1 - self._input_keep_prob)

        with tf.name_scope('encoder'):
            encoded, states = self.__bi_lstm(encoded, num_layers=2, return_states=True)
            self.__context = attention(encoded)

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
            decoded = tf.nn.dropout(x, rate=1 - self._input_keep_prob)

        with tf.name_scope('decoder'):
            decoded = self.__bi_lstm(decoded, num_layers=2, return_states=False, initial_states=states)
            decoded = h_dropout(self.__projection(decoded))
            decoded = tf.keras.layers.Dense(units=self._dims, activation=tf.nn.relu)(decoded)

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

            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
            opt = tf.compat.v1.train.AdamOptimizer(self._lr, epsilon=1e-2)

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

    def __bi_lstm(self, encoder_input, num_layers=2, return_states=False, initial_states=None):
        params = dict(
            units=self._num_hidden,
            return_sequences=True,
            return_state=return_states,
            dtype=tf.float32
        )
        lstm = tf.keras.layers.CuDNNLSTM if self.__use_gpu else tf.keras.layers.LSTM

        with tf.name_scope('bi-lstm'):
            for layer in range(num_layers):
                fwd_inputs = dict(inputs=encoder_input)
                bwd_inputs = dict(inputs=encoder_input)
                if initial_states is not None and layer == 0:
                    assert isinstance(initial_states, list) and len(initial_states) == 2
                    fwd_inputs["initial_state"] = initial_states[0]
                    bwd_inputs["initial_state"] = initial_states[1]
                if return_states:
                    decode_forward, *forward_state = lstm(**params)(**fwd_inputs)
                    decode_backward, *backward_state = lstm(**params, go_backwards=True)(**bwd_inputs)
                else:
                    decode_forward = lstm(**params)(**fwd_inputs)
                    decode_backward = lstm(**params, go_backwards=True)(**bwd_inputs)
                encoder_input = tf.keras.layers.Concatenate()([decode_forward, decode_backward])
            if return_states:
                return encoder_input, [forward_state, backward_state]
            return encoder_input

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

    def generalized_cosine(self, in_seq, out_seq):
        """
        Computes the sentence similarity loss function
        :param in_seq: (tf.Tensor)
        :param out_seq: (tf.Tensor)
        :return: (tf.float32)
        """

        epsilon = tf.constant(1e-8, dtype=tf.float32, shape=[1])
        length = tf.cast(self._seq_lengths, dtype=tf.float32)

        # this is a mask on the cosine similarity along the time-steps axis for each example
        mask = tf.sequence_mask(lengths=length, maxlen=self._time_steps, name='cosine_mask')
        mask = tf.cast(mask, dtype=tf.float32)

        # regularize the length with an epsilon for 0-length sequences
        length = tf.add(length, epsilon)

        inner = tf.reduce_sum(tf.multiply(in_seq, out_seq), axis=2)
        similarity = tf.multiply(1 - inner, mask)  # row-wise cosine similarity
        similarity = tf.reduce_sum(similarity, axis=1)
        # this, along with the line above computes the average similarity for all time steps in an example
        similarity = tf.divide(similarity, length)

        return tf.reduce_mean(similarity)

    def __cosine_cost(self):
        with tf.name_scope('cosine_similarity_loss'):
            loss = self.generalized_cosine(
                in_seq=tf.nn.l2_normalize(self._input, axis=2),
                out_seq=tf.nn.l2_normalize(self._output, axis=2)
            )

        with tf.name_scope('l2_loss'):
            weights = tf.trainable_variables()

            # only perform L2-regularization on the fully connected layer(s)
            l2_losses = [tf.nn.l2_loss(v) for v in weights if 'dense_weight' in v.name]
            loss += 1e-2 * tf.add_n(l2_losses)
        return loss

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
    def lr(self):
        return self._lr

    @property
    def clip_norm(self):
        return self._clip_norm

    @property
    def embedding(self):
        return self.__context
