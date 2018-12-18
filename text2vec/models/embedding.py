import tensorflow as tf
from . import model_utils as mu


class TextAttention(object):
    """
    This is a seq2seq model that uses a modified attention mechanism
    that is intended to relieve the burden on the dense output layer, and
    allow the decoder to focus on the most relevant learned parts of the
    input.

    I should make special note that this is a non-traditional implementation
    of attention mechanism (probably should not be called 'attention' in this context)
    , in that I am only using the source sequence information to learn the context
    vectors, and then using the context vectors onto which I project the target sequence.
    """

    def __init__(self, max_sequence_len, vocab_size, embedding_size, num_hidden, attention_size, is_training=False):
        self.seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len], name='input')
        self.keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

        self.__use_gpu = tf.test.is_gpu_available()

        self._batch_size, self._time_steps = self.seq_input.get_shape().as_list()
        self._dims = embedding_size
        self._num_labels = vocab_size
        self._num_hidden = num_hidden
        self._attention_size = attention_size

        self._input_keep_prob, self._lstm_keep_prob, self._dense_keep_prob = tf.unstack(self.keep_prob)

        # input embedding
        with tf.variable_scope('embedding'):
            self._seq_lengths = tf.count_nonzero(self.seq_input, axis=1, name="sequence_lengths")
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, self._dims], -1.0, 1.0),
                name="embeddings",
                dtype=tf.float32,
                trainable=True
            )
            self._input = tf.nn.embedding_lookup(embeddings, self.seq_input)
            x = self._input_op()

        # bi-directional encoder
        self.context, final_state, shape = self._encoder(x)

        if is_training:
            # bi-directional decoder, projects onto the context vectors
            decoded = self._decoder(tensor_shape=shape, initial_state=final_state)

            # dense layer
            self._output = self._dense(decoded, category_output=True)
            self._input = tf.stack(self._input)

            with tf.variable_scope('cost'):
                # self.loss = self._cost()

                target = tf.concat([tf.zeros_like(self.seq_input[:, :1]), self.seq_input[:, :-1]], axis=1)
                self.loss = self.__seq_cost(target_sequences=target, sequence_logits=self._output)

            with tf.variable_scope('optimizer'):
                self._lr = tf.Variable(0.0, trainable=False)
                self._clip_norm = tf.Variable(0.0, trainable=False)
                t_vars = tf.trainable_variables()

                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), self._clip_norm)
                opt = tf.train.AdamOptimizer(self._lr, epsilon=1e-2)

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

    def _input_op(self):
        with tf.variable_scope('input_dropout'):
            input_x = self._input
            return tf.nn.dropout(x=input_x, keep_prob=self._input_keep_prob)

    def _encoder(self, input_x):
        with tf.variable_scope('encoder'):
            num_layers = 2

            forward = mu.build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self.__use_gpu
            )
            backward = mu.build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self.__use_gpu
            )

            output_seq, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward,
                cell_bw=backward,
                inputs=input_x,
                dtype=tf.float32,
                sequence_length=self._seq_lengths
            )

            # perform element-wise summations to combine the forward and backward sequences
            encoder_output = mu.concat_reducer(seq_fw=output_seq[0], seq_bw=output_seq[1])
            encoder_state = final_state

            # get the context vectors from the attention mechanism
            context = self._attention(input_x=encoder_output)

            return context, encoder_state, tf.shape(encoder_output)

    def _attention(self, input_x):
        with tf.variable_scope('source_attention'):
            in_dim = input_x.get_shape().as_list()[-1]
            w_omega = tf.get_variable(
                "w_omega",
                shape=[in_dim, self._attention_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True)
            )
            b_omega = tf.get_variable("b_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())
            u_omega = tf.get_variable("u_omega", shape=[self._attention_size], initializer=tf.zeros_initializer())

            v = tf.tanh(tf.einsum("ijk,kl->ijl", input_x, w_omega) + b_omega)
            vu = tf.einsum("ijl,l->ij", v, u_omega, name="Bahdanau_score")
            alphas = tf.nn.softmax(vu, name="attention_weights")

            output = tf.reduce_sum(input_x * tf.expand_dims(alphas, -1), 1, name="context_vector")
            return output

    def _decoder(self, tensor_shape, initial_state):
        with tf.variable_scope('decoder'):
            num_layers = 2

            forward = mu.build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self.__use_gpu
            )
            backward = mu.build_cell(
                num_layers=num_layers,
                num_hidden=self._num_hidden,
                keep_prob=self._lstm_keep_prob,
                use_cuda=self.__use_gpu
            )

            dec_inputs = tf.zeros(shape=tensor_shape, dtype=tf.float32)
            dec_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward,
                cell_bw=backward,
                inputs=dec_inputs,
                initial_state_fw=initial_state[0],
                initial_state_bw=initial_state[1],
                dtype=tf.float32,
                sequence_length=self._seq_lengths
            )
            dec_outputs = mu.concat_reducer(seq_fw=dec_outputs[0], seq_bw=dec_outputs[1])
            return dec_outputs

    def _dense(self, input_x, category_output=False):
        with tf.variable_scope('dense'):
            input_x = tf.nn.dropout(input_x, keep_prob=self._dense_keep_prob, name="dense_dropout")

            with tf.variable_scope('projection'):
                inner_product = tf.einsum("ijk,ik->ij", input_x, self.context)
                context_norm_sqrd = tf.norm(self.context, axis=1) ** 2

                # to make this work on the GPU we can't do broadcasting
                # so we need to expand context norm to a 2D tensor,
                # then tile that out to have the same number of columns as inner product
                context_norm_sqrd = tf.tile(tf.expand_dims(context_norm_sqrd, -1), [1, self._time_steps])

                alpha = tf.divide(inner_product, context_norm_sqrd)
                projection = tf.einsum("ij,ik->ijk", alpha, self.context, name="projection_op")

            if category_output:
                output_ = tf.layers.dense(inputs=projection, units=self._num_labels)
            else:
                output_ = tf.layers.dense(inputs=projection, units=self._dims)
            return output_

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

    def _cost(self):
        with tf.variable_scope('cosine_similarity_loss'):
            loss = self.generalized_cosine(
                in_seq=tf.nn.l2_normalize(self._input, axis=2),
                out_seq=tf.nn.l2_normalize(self._output, axis=2)
            )

        with tf.variable_scope('l2_loss'):
            weights = tf.trainable_variables()

            # only perform L2-regularization on the fully connected layer(s)
            l2_losses = [tf.nn.l2_loss(v) for v in weights if 'dense_weight' in v.name]
            loss += 1e-2 * tf.add_n(l2_losses)
        return loss

    def __seq_cost(self, target_sequences, sequence_logits, smoothing=False):
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
    def lr(self):
        return self._lr

    @property
    def clip_norm(self):
        return self._clip_norm

    @property
    def embedding(self):
        return self.context
