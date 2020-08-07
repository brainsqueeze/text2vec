from .components.attention import BahdanauAttention
from .components.recurrent import BidirectionalLSTM
from .components.utils import TensorProjection
import tensorflow as tf


class RecurrentEncoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_hidden, num_layers=2, input_keep_prob=1.0, **kwargs):
        super(RecurrentEncoder, self).__init__()
        self.max_sequence_length = max_sequence_len

        self.drop = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=True)
        self.attention = BahdanauAttention(size=2 * num_hidden)

    def __call__(self, x, mask, training=False, **kwargs):
        with tf.name_scope("RecurrentEncoder"):
            x = self.drop(x, training=training)
            x, states = self.bi_lstm(x)
            context = self.attention(x)

            if training:
                return x, context, states
            return context


class RecurrentDecoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_labels, num_hidden, embedding_size=50, num_layers=2,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(RecurrentDecoder, self).__init__()
        self.max_sequence_length = max_sequence_len
        dims = embedding_size

        self.drop = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_drop = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.projection = TensorProjection()

        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=False)
        self.dense = tf.keras.layers.Dense(units=dims, activation=tf.nn.relu)

    def __call__(self, x_enc, enc_mask, x_dec, dec_mask, context, attention, training=False, **kwargs):
        with tf.name_scope("RecurrentDecoder"):
            initial_state = kwargs.get("initial_state")
            x = self.drop(x_dec, training=training)
            if initial_state is not None:
                x = self.bi_lstm(x, initial_states=initial_state[0])
            else:
                x = self.bi_lstm(x)
            x = self.h_drop(self.projection(x, projection_vector=context))
            x = self.dense(x)
            return x
