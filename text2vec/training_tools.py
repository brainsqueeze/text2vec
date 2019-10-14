from text2vec.models import TransformerEncoder
from text2vec.models import TransformerDecoder
from text2vec.models import InputFeeder

from text2vec.models.components.utils import sequence_cost

import tensorflow as tf


def processing_encoding_input(tokens, max_sequence_length, inputs_handler):
    assert isinstance(tokens, tf.RaggedTensor)
    assert isinstance(inputs_handler, InputFeeder)

    x, mask, _ = inputs_handler(tokens, max_sequence_length=max_sequence_length)
    return x, mask


def process_decoding_input(tokens, max_sequence_length, inputs_handler):
    assert isinstance(tokens, tf.RaggedTensor)
    assert isinstance(inputs_handler, InputFeeder)

    batch_size = tokens.nrows()
    bos = tf.fill([batch_size], value='<s>')
    bos = tf.expand_dims(bos, axis=-1)
    eos = tf.fill([batch_size], value='</s>')
    eos = tf.expand_dims(eos, axis=-1)

    target = tf.concat([tokens, eos], axis=1)
    target = tf.ragged.map_flat_values(inputs_handler.table.lookup, target)
    target = target[:, :max_sequence_length]

    dec_tokens = tf.concat([bos, tokens], axis=-1)
    x, mask, time_steps = inputs_handler(dec_tokens, max_sequence_length=max_sequence_length)
    return x, mask, time_steps, target


@tf.function
def train_step(sentences, inputs_handler, encoder, decoder, optimizer):
    assert isinstance(inputs_handler, InputFeeder)
    assert isinstance(encoder, TransformerEncoder)
    assert isinstance(decoder, TransformerDecoder)

    with tf.GradientTape() as tape:
        tokens = tf.string_split(sentences, sep=' ')
        tokens = tf.RaggedTensor.from_sparse(tokens)

        x_enc, enc_mask = processing_encoding_input(
            tokens=tokens,
            max_sequence_length=encoder.max_sequence_length,
            inputs_handler=inputs_handler
        )
        x_dec, dec_mask, dec_time_steps, targets = process_decoding_input(
            tokens=tokens,
            max_sequence_length=decoder.max_sequence_length,
            inputs_handler=inputs_handler
        )

        x_enc, context = encoder(x_enc, mask=enc_mask, training=True)
        x_out = decoder(
            x_enc=x_enc,
            enc_mask=enc_mask,
            x_dec=x_dec,
            dec_mask=dec_mask,
            context=context,
            attention=encoder.attention,
            embeddings=inputs_handler.embeddings
        )
        x_out = x_out[:, :dec_time_steps]
        targets = targets.to_tensor(default_value=0)
        loss = sequence_cost(
            target_sequences=targets,
            sequence_logits=x_out,
            num_labels=inputs_handler.num_labels,
            smoothing=False
        )

    trainable_variables = tf.trainable_variables(encoder)
    gradients = tape.gradient()
    return x_out
