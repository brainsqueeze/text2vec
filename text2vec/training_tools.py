from text2vec.models import InputFeeder
# from text2vec.models.components.utils import sequence_cost
import tensorflow as tf


class EncodingModel(tf.keras.Model):

    def __init__(self, feeder, encoder, decoder):
        super(EncodingModel, self).__init__()
        assert isinstance(feeder, InputFeeder)
        assert isinstance(encoder, tf.keras.layers.Layer)
        assert isinstance(decoder, tf.keras.layers.Layer)

        self.embed_layer = feeder
        self.encode_layer = encoder
        self.decode_layer = decoder

        self.num_labels = feeder.num_labels

    def process_inputs(self, tokens, encoding=True):
        assert isinstance(tokens, tf.RaggedTensor)

        if encoding:
            x, mask, _ = self.embed_layer(tokens, max_sequence_length=self.encode_layer.max_sequence_length)
            return x, mask

        batch_size = tokens.nrows()

        with tf.name_scope('targets'):
            eos = tf.fill([batch_size], value='</s>', name='eos-tag')
            eos = tf.expand_dims(eos, axis=-1, name='eos-tag-expand')

            target = tf.concat([tokens, eos], axis=1, name='eos-concat')
            target = tf.ragged.map_flat_values(self.embed_layer.table.lookup, target)
            target = target[:, :self.decode_layer.max_sequence_length]

        with tf.name_scope('decode-tokens'):
            bos = tf.fill([batch_size], value='<s>', name='bos-tag')
            bos = tf.expand_dims(bos, axis=-1, name='bos-tag-expand')

            dec_tokens = tf.concat([bos, tokens], axis=-1, name='bos-concat')
        x, mask, time_steps = self.embed_layer(dec_tokens, max_sequence_length=self.decode_layer.max_sequence_length)
        return x, mask, time_steps, target

    def __call__(self, sentences, training=False, **kwargs):
        # turn sentences into ragged tensors of tokens
        tokens = tf.strings.split(sentences, sep=' ')

        # turn incoming sentences into relevant tensor batches
        with tf.name_scope('Encoding'):
            x_enc, enc_mask = self.process_inputs(tokens)
            if not training:
                return self.encode_layer((x_enc, enc_mask), training=False)
            x_enc, context = self.encode_layer((x_enc, enc_mask), training=True)

        with tf.name_scope('Decoding'):
            x_dec, dec_mask, dec_time_steps, targets = self.process_inputs(tokens, encoding=False)
            x_out = self.decode_layer((
                x_enc,
                enc_mask,
                x_dec,
                dec_mask,
                context,
                self.encode_layer.attention,
                self.embed_layer.embeddings
            ), training=training)

        return x_out, dec_time_steps, targets.to_tensor(default_value=0)


def sequence_cost(target_sequences, sequence_logits, num_labels, smoothing=False):
    with tf.name_scope('Cost'):
        if smoothing:
            smoothing = 0.1
            targets = tf.one_hot(target_sequences, depth=num_labels, on_value=1.0, off_value=0.0, axis=-1)
            loss = tf.losses.softmax_cross_entropy(
                logits=sequence_logits,
                onehot_labels=targets,
                label_smoothing=smoothing,
                reduction=tf.losses.Reduction.NONE
            )
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sequence_logits, labels=target_sequences)

        loss = tf.reduce_mean(loss)
        return loss
