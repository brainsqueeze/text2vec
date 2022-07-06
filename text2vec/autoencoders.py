# pylint: disable=too-many-ancestors
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers, Model

from text2vec.models.components.feeder import Tokenizer
from text2vec.models.components.text_inputs import TokenEmbed
from text2vec.models.components.text_inputs import Embed
from text2vec.models.transformer import TransformerEncoder
from text2vec.models.transformer import TransformerDecoder
from text2vec.models.sequential import RecurrentEncoder
from text2vec.models.sequential import RecurrentDecoder


class TransformerAutoEncoder(Model):
    """Wrapper model class to combine the transformer based encoder-decoder training pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    embedding_size : int
        Dimensionality of the word-embeddings.
    token_hash : dict, optional
        Token -> integer vocabulary lookup, by default None
    vocab_size : int, optional
        Size of the vocabulary. Set this if pre-computing token IDs to pass to the model, by default None
    unknown_token : str, optional
        The placeholder value for OOV terms, by default '<unk>'
    sep : str, optional
        Token separator by default ' '
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    hidden_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

    Raises
    ------
    ValueError
        Raised if neither a vocab dictionary  or a vocab size is provided.


    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.autoencoders import TransformerAutoEncoder

    lookup = {'string': 0, 'is': 1, 'example': 2, '<unk>': 3}
    model = TransformerAutoEncoder(token_hash=lookup, max_sequence_len=10, embedding_size=16)
    text = tf.constant(["sample string .", "this is a second example ."])
    encoded, context_vectors = model(text)
    ```
    """

    def __init__(self, max_sequence_len: int, embedding_size: int,
                 token_hash: Optional[dict] = None, vocab_size: Optional[int] = None,
                 unknown_token: str = '<unk>', sep: str = ' ',
                 input_drop_rate: float = 0, hidden_drop_rate: float = 0):
        super().__init__()

        if token_hash is None and vocab_size is None:
            raise ValueError("Must provide either a dictionary mapping or a dictionary size if using token IDs")

        params = dict(
            max_sequence_len=max_sequence_len,
            embedding_size=embedding_size,
            input_drop_rate=input_drop_rate,
            hidden_drop_rate=hidden_drop_rate
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
            self.tokenizer = layers.Lambda(lambda x: x)  # this is only for consistency, identity map
            self.embed_layer = Embed(
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                max_sequence_len=max_sequence_len
            )

        self.encode_layer = TransformerEncoder(n_stacks=1, num_layers=8, **params)
        self.decode_layer = TransformerDecoder(n_stacks=1, num_layers=8, **params)

    def call(self, tokens, training: bool = False):  # pylint: disable=missing-function-docstring
        tokens = self.tokenizer(tokens)
        x_enc, enc_mask, _ = self.embed_layer(tokens, training=training)
        x_enc, context = self.encode_layer(x_enc, mask=enc_mask, training=training)
        return x_enc, context, enc_mask

    def train_step(self, data):  # pylint: disable=missing-function-docstring
        encoding_tok, decoding_tok = data
        decoding_tok = self.tokenizer(decoding_tok)

        with tf.GradientTape() as tape:
            with tf.name_scope('Encoding'):
                x_enc, context, enc_mask = self(encoding_tok, training=True)

            with tf.name_scope('Decoding'):
                targets = decoding_tok[:, 1:]  # skip the <s> token with the slice on axis=1
                if isinstance(self.embed_layer, TokenEmbed):
                    targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, targets)
                targets = self.embed_layer.slicer(targets)

                decoding_tok, dec_mask, _ = self.embed_layer(decoding_tok[:, :-1])  # skip </s>
                decoding_tok = self.decode_layer(
                    x_enc=x_enc,
                    x_dec=decoding_tok,
                    dec_mask=dec_mask,
                    context=context,
                    attention=self.encode_layer.attention,
                    training=True
                )
                decoding_tok = tf.tensordot(decoding_tok, self.embed_layer.embeddings, axes=[2, 1])

            loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decoding_tok,
                labels=targets.to_tensor(default_value=0)
            )
            loss = loss * dec_mask
            loss = tf.math.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if hasattr(self.optimizer, 'learning_rate') and callable(self.optimizer.learning_rate):
            return {"loss": loss, 'learning_rate': self.optimizer.learning_rate(self.optimizer.iterations)}
        return {"loss": loss, 'learning_rate': self.optimizer.learning_rate}

    def test_step(self, data):  # pylint: disable=missing-function-docstring
        encoding_tok, decoding_tok = data
        decoding_tok = self.tokenizer(decoding_tok)

        with tf.name_scope('Encoding'):
            x_enc, context, enc_mask = self(encoding_tok, training=False)

        with tf.name_scope('Decoding'):
            targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, decoding_tok[:, 1:])  # skip <s>
            targets = self.embed_layer.slicer(targets)

            decoding_tok, dec_mask, _ = self.embed_layer(decoding_tok[:, :-1])  # skip </s>
            decoding_tok = self.decode_layer(
                x_enc=x_enc,
                enc_mask=enc_mask,
                x_dec=decoding_tok,
                dec_mask=dec_mask,
                context=context,
                attention=self.encode_layer.attention,
                training=False
            )
            decoding_tok = tf.tensordot(decoding_tok, self.embed_layer.embeddings, axes=[2, 1])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=decoding_tok,
            labels=targets.to_tensor(default_value=0)
        )
        loss = loss * dec_mask
        loss = tf.math.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)
        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences) -> Dict[str, tf.Tensor]:
        """Takes batches of free text and returns context vectors for each example.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        Dict[str, tf.Tensor]
            Attention vector and hidden state sequences with shapes (batch_size, embedding_size)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        sequences, attention, _ = self(sentences, training=False)
        return {"sequences": sequences, "attention": attention}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences) -> Dict[str, tf.Tensor]:
        """Takes batches of free text and returns word embeddings along with the associate token.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        Dict[str, tf.Tensor]
            Padded tokens and embedding vectors with shapes (batch_size, max_sequence_len)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        tokens = self.tokenizer(sentences)
        return {
            "tokens": tokens.to_tensor('</>'),
            "embeddings": self.embed_layer.get_embedding(tokens).to_tensor(0)
        }


class LstmAutoEncoder(Model):
    """Wrapper model class to combine the LSTM based encoder-decoder training pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    embedding_size : int
        Dimensionality of the word-embeddings.
    num_hidden : int, optional
        Size of the hidden LSTM state, by default 64
    token_hash : dict, optional
        Token -> integer vocabulary lookup, by default None
    vocab_size : int, optional
        Size of the vocabulary. Set this if pre-computing token IDs to pass to the model, by default None
    unknown_token : str, optional
        The placeholder value for OOV terms, by default '<unk>'
    sep : str, optional
        Token separator by default ' '
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    hidden_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

    Raises
    ------
    ValueError
        Raised if neither a vocab dictionary  or a vocab size is provided.


    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.autoencoders import LstmAutoEncoder

    lookup = {'string': 0, 'is': 1, 'example': 2, '<unk>': 3}
    model = LstmAutoEncoder(token_hash=lookup, max_sequence_len=10, embedding_size=16)
    text = tf.constant(["sample string .", "this is a second example ."])
    encoded, context_vectors = model(text)
    ```
    """

    def __init__(self, max_sequence_len: int, embedding_size: int, num_hidden: int = 64,
                 token_hash: Optional[dict] = None, vocab_size: Optional[int] = None,
                 unknown_token: str = '<unk>', sep: str = ' ',
                 input_drop_rate: float = 0., hidden_drop_rate: float = 0.):
        super().__init__()

        if token_hash is None and vocab_size is None:
            raise ValueError("Must provide either a dictionary mapping or a dictionary size if using token IDs")

        params = dict(
            max_sequence_len=max_sequence_len,
            embedding_size=embedding_size,
            input_drop_rate=input_drop_rate,
            hidden_drop_rate=hidden_drop_rate
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
            self.tokenizer = layers.Lambda(lambda x: x)  # this is only for consistency, identity map
            self.embed_layer = Embed(
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                max_sequence_len=max_sequence_len
            )

        self.encode_layer = RecurrentEncoder(num_hidden=num_hidden, **params)
        self.decode_layer = RecurrentDecoder(num_hidden=num_hidden, **params)

    def call(self, tokens, training: bool = False):  # pylint: disable=missing-function-docstring
        tokens = self.tokenizer(tokens)
        x_enc, enc_mask, _ = self.embed_layer(tokens, training=training)
        x_enc, context, *states = self.encode_layer(x_enc, mask=enc_mask, training=training)
        return x_enc, context, enc_mask, states

    def train_step(self, data):  # pylint: disable=missing-function-docstring
        encoding_tok, decoding_tok = data
        decoding_tok = self.tokenizer(decoding_tok)

        with tf.GradientTape() as tape:
            with tf.name_scope('Encoding'):
                x_enc, context, enc_mask, states = self(encoding_tok, training=True)

            with tf.name_scope('Decoding'):
                targets = decoding_tok[:, 1:]  # skip the <s> token with the slice on axis=1
                if isinstance(self.embed_layer, TokenEmbed):
                    targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, targets)
                targets = self.embed_layer.slicer(targets)

                decoding_tok, dec_mask, _ = self.embed_layer(decoding_tok[:, :-1])
                decoding_tok = self.decode_layer(
                    x_enc=x_enc,
                    enc_mask=enc_mask,
                    x_dec=decoding_tok,
                    dec_mask=dec_mask,
                    context=context,
                    initial_state=states,
                    training=True
                )
                decoding_tok = tf.tensordot(decoding_tok, self.embed_layer.embeddings, axes=[2, 1])

            loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=decoding_tok,
                labels=targets.to_tensor(default_value=0)
            )
            loss = loss * dec_mask
            loss = tf.math.reduce_sum(loss, axis=1)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if hasattr(self.optimizer, 'learning_rate') and callable(self.optimizer.learning_rate):
            return {"loss": loss, 'learning_rate': self.optimizer.learning_rate(self.optimizer.iterations)}
        return {"loss": loss, 'learning_rate': self.optimizer.learning_rate}

    def test_step(self, data):  # pylint: disable=missing-function-docstring
        encoding_tok, decoding_tok = data
        decoding_tok = self.tokenizer(decoding_tok)

        with tf.name_scope('Encoding'):
            x_enc, context, enc_mask, states = self(encoding_tok, training=False)

        with tf.name_scope('Decoding'):
            targets = decoding_tok[:, 1:]  # skip the <s> token with the slice on axis=1
            if isinstance(self.embed_layer, TokenEmbed):
                targets = tf.ragged.map_flat_values(self.embed_layer.table.lookup, targets)
            targets = self.embed_layer.slicer(targets)

            decoding_tok, dec_mask, _ = self.embed_layer(decoding_tok[:, :-1])
            decoding_tok = self.decode_layer(
                x_enc=x_enc,
                enc_mask=enc_mask,
                x_dec=decoding_tok,
                dec_mask=dec_mask,
                context=context,
                initial_state=states,
                training=False
            )
            decoding_tok = tf.tensordot(decoding_tok, self.embed_layer.embeddings, axes=[2, 1])

        loss = loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=decoding_tok,
            labels=targets.to_tensor(default_value=0)
        )
        loss = loss * dec_mask
        loss = tf.math.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)

        return {"loss": loss, **{m.name: m.result() for m in self.metrics}}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences) -> Dict[str, tf.Tensor]:
        """Takes batches of free text and returns context vectors for each example.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        Dict[str, tf.Tensor]
            Attention vector and hidden state sequences with shapes (batch_size, embedding_size)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        sequences, attention, *_ = self(sentences, training=False)
        return {"sequences": sequences, "attention": attention}

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences) -> Dict[str, tf.Tensor]:
        """Takes batches of free text and returns word embeddings along with the associate token.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        Dict[str, tf.Tensor]
            Padded tokens and embedding vectors with shapes (batch_size, max_sequence_len)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        tokens = self.tokenizer(sentences)
        return {
            "tokens": tokens.to_tensor('</>'),
            "embeddings": self.embed_layer.get_embedding(tokens).to_tensor(0)
        }
