from .components.attention import BahdanauAttention
from .components.recurrent import BidirectionalLSTM
from .components import utils

import tensorflow as tf


class RecurrentEncoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_hidden, num_layers=2, input_keep_prob=1.0, **kwargs):
        super(RecurrentEncoder, self).__init__()
        self.max_sequence_length = max_sequence_len

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=True)
        self.attention = BahdanauAttention(size=2 * num_hidden)

    def __call__(self, x, mask, training=False, **kwargs):
        x = self.dropout(x, training=training)
        x, states = self.bi_lstm(x)
        context = self.attention(x)

        if training:
            return x, context
        return context


class RecurrentDecoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_labels, num_hidden, embedding_size=50, num_layers=2,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(RecurrentDecoder, self).__init__()
        self.max_sequence_length = max_sequence_len
        dims = embedding_size

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_dropout = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.projection = utils.TensorProjection()

        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=False)
        self.dense = tf.keras.layers.Dense(units=dims, activation=tf.nn.relu)
        self.bias = tf.Variable(tf.zeros([num_labels]), name='bias', dtype=tf.float32, trainable=True)

    def __call__(self, x_enc, enc_mask, x_dec, dec_mask, context, attention, embeddings, training=False, **kwargs):
        x = self.dropout(x_dec, training=training)
        x = self.bi_lstm(x)
        x = self.h_dropout(self.projection(x, projection_vector=context))
        x = self.dense(x)
        return tf.tensordot(x, embeddings, axes=[2, 1]) + self.bias
