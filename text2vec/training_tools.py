from text2vec.models import TextInput
from text2vec.models import Tokenizer
from text2vec.models import TransformerEncoder
from text2vec.models import TransformerDecoder
from text2vec.models import RecurrentEncoder
from text2vec.models import RecurrentDecoder
import tensorflow as tf


class EncodingModel(tf.keras.Model):

    def __init__(self, token_hash, max_sequence_len, n_stacks=1, layers=8, num_hidden=64,
                 input_keep_prob=1.0, hidden_keep_prob=1.0, embedding_size=64, recurrent=False, sep=' '):
        super(EncodingModel, self).__init__()

        params = dict(
            max_sequence_len=max_sequence_len,
            embedding_size=embedding_size,
            input_keep_prob=input_keep_prob,
            hidden_keep_prob=hidden_keep_prob
        )
        self.embed_layer = TextInput(token_hash=token_hash, **params)
        self.tokenizer = Tokenizer(sep)
        num_labels = len(token_hash) + 1

        if recurrent:
            self.encode_layer = RecurrentEncoder(num_hidden=num_hidden, **params)
            self.decode_layer = RecurrentDecoder(num_hidden=num_hidden, num_labels=num_labels, **params)
        else:
            self.encode_layer = TransformerEncoder(n_stacks=n_stacks, layers=layers, **params)
            self.decode_layer = TransformerDecoder(n_stacks=n_stacks, layers=layers, num_labels=num_labels, **params)

    def __call__(self, sentences, training=False, return_vectors=False):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens

        # turn incoming sentences into relevant tensor batches
        with tf.name_scope('Encoding'):
            x_enc, enc_mask, _ = self.embed_layer(tokens)
            if not training:
                return self.encode_layer(x_enc, mask=enc_mask, training=False)
            x_enc, context = self.encode_layer(x_enc, mask=enc_mask, training=True)

        with tf.name_scope('Decoding'):
            batch_size = tokens.nrows()

            with tf.name_scope('targets'):
                eos = tf.fill([batch_size], value='</s>', name='eos-tag')
                eos = tf.expand_dims(eos, axis=-1, name='eos-tag-expand')

                targets = tf.concat([tokens, eos], axis=1, name='eos-concat')
                targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, targets)
                targets = self.embed_layer.slicer(targets)

            with tf.name_scope('decode-tokens'):
                bos = tf.fill([batch_size], value='<s>', name='bos-tag')
                bos = tf.expand_dims(bos, axis=-1, name='bos-tag-expand')

                dec_tokens = tf.concat([bos, tokens], axis=-1, name='bos-concat')
            x_dec, dec_mask, dec_time_steps = self.embed_layer(dec_tokens)
            x_out = self.decode_layer(
                x_enc=x_enc,
                enc_mask=enc_mask,
                x_dec=x_dec,
                dec_mask=dec_mask,
                context=context,
                attention=self.encode_layer.attention,
                embeddings=self.embed_layer.embeddings,
                training=training
            )

        if return_vectors:
            return x_out, dec_time_steps, targets.to_tensor(default_value=0), context
        return x_out, dec_time_steps, targets.to_tensor(default_value=0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        x_enc, enc_mask, _ = self.embed_layer(tokens)
        return self.encode_layer(x_enc, mask=enc_mask, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        return tokens.to_tensor(''), self.embed_layer(tokens, output_embeddings=True).to_tensor(0)


class ServingModel(tf.keras.Model):

    def __init__(self, embed_layer, encode_layer, sep=' '):
        super(ServingModel, self).__init__()

        assert isinstance(embed_layer, TextInput)
        assert type(encode_layer) in {RecurrentEncoder, TransformerEncoder}

        self.embed_layer = embed_layer
        self.tokenizer = Tokenizer(sep)
        self.encode_layer = encode_layer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        x_enc, enc_mask, _ = self.embed_layer(tokens)
        return self.encode_layer(x_enc, mask=enc_mask, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        return tokens.to_tensor(''), self.embed_layer(tokens, output_embeddings=True).to_tensor(0)


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


def vector_cost(context_vectors):
    with tf.name_scope('VectorCost'):
        rows = tf.shape(context_vectors)[0]
        context_vectors = tf.linalg.l2_normalize(context_vectors, axis=-1)
        cosine = tf.tensordot(context_vectors, tf.transpose(context_vectors), axes=[1, 0])
        identity = tf.eye(rows)
        return tf.reduce_mean((identity - cosine) ** 2)
