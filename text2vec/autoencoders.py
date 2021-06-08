import tensorflow as tf

from text2vec.models.components.feeder import Tokenizer
from text2vec.models.components.text_inputs import TokenEmbed
from text2vec.models.components.text_inputs import Embed
from text2vec.models.transformer import TransformerEncoder
from text2vec.models.transformer import TransformerDecoder


class TransformerAutoEncoder(tf.keras.Model):

    def __init__(self, max_sequence_len: int, embedding_size: int,
                 token_hash: dict = None, vocab_size: int = None, unknown_token: str = '<unk>', sep: int = ' ',
                 input_keep_prob: float = 1.0, hidden_keep_prob: float = 1.0):
        super().__init__()

        if token_hash is None and vocab_size is None:
            raise ValueError("Must provide either a dictionary mapping or a dictionary size if using token IDs")

        params = dict(
            max_sequence_len=max_sequence_len,
            embedding_size=embedding_size,
            input_keep_prob=input_keep_prob,
            hidden_keep_prob=hidden_keep_prob
        )

        if token_hash is not None:
            self.tokenizer = Tokenizer(sep)
            self.embed_layer = TokenEmbed(
                token_hash=token_hash,
                embedding_size=embedding_size,
                max_sequence_len=max_sequence_len,
                unknown_token=unknown_token
            )
        else:
            self.tokenizer = tf.keras.layers.Lambda(lambda x: x)  # this is only for consistency, identity map
            self.embed_layer = Embed(
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                max_sequence_len=max_sequence_len
            )

        self.encode_layer = TransformerEncoder(n_stacks=1, layers=8, **params)
        self.decode_layer = TransformerDecoder(n_stacks=1, layers=8, **params)

    def train_step(self, data):
        encoding_tok, decoding_tok = data
        encoding_tok = self.tokenizer(encoding_tok)
        decoding_tok = self.tokenizer(decoding_tok)

        with tf.GradientTape() as tape:
            with tf.name_scope('Encoding'):
                x_enc, enc_mask, _ = self.embed_layer(encoding_tok)
                x_enc, context = self.encode_layer(x_enc, mask=enc_mask, training=True)

            with tf.name_scope('Decoding'):
                targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, encoding_tok)
                targets = self.embed_layer.slicer(targets)

                x_dec, dec_mask, dec_time_steps = self.embed_layer(decoding_tok)
                x_dec = self.decode_layer(
                    x_enc=x_enc,
                    enc_mask=enc_mask,
                    x_dec=x_dec,
                    dec_mask=dec_mask,
                    context=context,
                    attention=self.encode_layer.attention,
                    training=True
                )
                x_dec = tf.tensordot(x_dec, self.embed_layer.embeddings, axes=[2, 1])

            loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=x_dec[:, :dec_time_steps],
                labels=targets.to_tensor(default_value=0)
            )
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if hasattr(self.optimizer, 'learning_rate') and callable(self.optimizer.learning_rate):
            return {"loss": loss, 'learning_rate': self.optimizer.learning_rate(self.optimizer.iterations)}
        return {"loss": loss}

    def __call__(self, tokens, **kwargs):
        tokens = self.tokenizer(tf.squeeze(tokens))
        x_enc, enc_mask, _ = self.embed_layer(tokens)
        return self.encode_layer(x_enc, mask=enc_mask, training=False)


class LstmAutoEncoder(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
