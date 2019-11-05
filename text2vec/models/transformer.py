from .components.attention import BahdanauAttention
from .components.attention import MultiHeadAttention
from .components.feed_forward import PositionWiseFFN
from .components import utils

import tensorflow as tf


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(TransformerEncoder, self).__init__()
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_dropout = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.layer_norm = utils.LayerNorm()

        self.positional_encode = utils.positional_encode(emb_dims=dims, max_sequence_length=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.attention = BahdanauAttention(size=dims)

    def __call__(self, inputs, training=False, **kwargs):
        x, mask = inputs
        x = self.dropout(x + (self.positional_encode * mask), training=training)

        for mha, ffn in zip(self.MHA, self.FFN):
            x = self.h_dropout(mha([x] * 3, training=training), training=training) + x
            x = self.layer_norm(x)
            x = self.h_dropout(ffn(x), training=training) + x
            x = self.layer_norm(x)

        context = self.attention((x, None))
        if training:
            return x, context
        return context


class TransformerDecoder(tf.keras.layers.Layer):

    def __init__(self, max_sequence_len, num_labels, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super(TransformerDecoder, self).__init__()
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.dropout = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_dropout = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.layer_norm = utils.LayerNorm()
        self.projection = utils.TensorProjection()

        self.positional_encode = utils.positional_encode(emb_dims=dims, max_sequence_length=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.bias = tf.Variable(tf.zeros([num_labels]), name='bias', dtype=tf.float32, trainable=True)

    def __call__(self, inputs, training=False, **kwargs):
        x_enc, enc_mask, x_dec, dec_mask, context, attention, embeddings = inputs

        x_dec = self.dropout(x_dec + (self.positional_encode * dec_mask), training=training)
        for mha, ffn in zip(self.MHA, self.FFN):
            x_dec = self.h_dropout(mha([x_dec] * 3, mask_future=True, training=training), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)

            cross_context = attention((x_enc * enc_mask, x_dec * dec_mask))
            x_dec = self.h_dropout(self.projection(x_dec, projection_vector=cross_context), training=training) + x_dec

            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_dropout(ffn(x_dec), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_dropout(self.projection(x_dec, projection_vector=context), training=training) + x_dec
        return tf.tensordot(x_dec, embeddings, axes=[2, 1]) + self.bias
