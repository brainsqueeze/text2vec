import tensorflow as tf

from text2vec.models import TextInput
from text2vec.models import Tokenizer
from text2vec.models import TransformerEncoder
from text2vec.models import TransformerDecoder
from text2vec.models import RecurrentEncoder
from text2vec.models import RecurrentDecoder


class EncodingModel(tf.keras.Model):
    """Wrapper model class to combine the encoder-decoder training pipeline.

        Parameters
        ----------
        token_hash : dict
            Token -> integer vocabulary lookup.
        max_sequence_len : int
            Longest sequence seen at training time.
        n_stacks : int, optional
            Number of encoding blocks to chain, by default 1
        layers : int, optional
            Number of layers in the multi-head-attention layer, by default 8
        num_hidden : int, optional
            Dimensionality of hidden LSTM layer weights, by default 64
        input_keep_prob : float, optional
            Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0
        hidden_keep_prob : float, optional
            Hidden states dropout. Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0
        embedding_size : int, optional
            Dimensionality of the word-embeddings, by default 64
        recurrent : bool, optional
            Set to True to use the LSTM based model, otherwise defaults to attention based model, by default False
        sep : str, optional
            Token separator, by default ' '

        Examples
        --------
        ```python
        import tensorflow as tf
        from text2vec.training_tools import EncodingModel

        lookup = {'string': 0, 'is': 1, 'example': 2}
        params = dict(
            max_sequence_len=10,
            embedding_size=16,
            input_keep_prob=0.9,
            hidden_keep_prob=0.75
        )
        model = EncodingModel(token_hash=lookup, layers=8, **params)

        text = tf.constant([
            "sample string .",
            "this is a second example ."
        ])
        y_hat, time_steps, targets, context_vectors = model(text, training=True, return_vectors=True)
        ```
        """

    def __init__(self, token_hash, max_sequence_len, n_stacks=1, layers=8, num_hidden=64,
                 input_keep_prob=1.0, hidden_keep_prob=1.0, embedding_size=64, recurrent=False, sep=' '):
        super().__init__()

        params = dict(
            max_sequence_len=max_sequence_len,
            embedding_size=embedding_size,
            input_keep_prob=input_keep_prob,
            hidden_keep_prob=hidden_keep_prob
        )
        self.embed_layer = TextInput(
            token_hash=token_hash,
            embedding_size=embedding_size,
            max_sequence_len=max_sequence_len
        )
        self.tokenizer = Tokenizer(sep)

        if recurrent:
            self.encode_layer = RecurrentEncoder(num_hidden=num_hidden, **params)
            self.decode_layer = RecurrentDecoder(num_hidden=num_hidden, **params)
        else:
            self.encode_layer = TransformerEncoder(n_stacks=n_stacks, layers=layers, **params)
            self.decode_layer = TransformerDecoder(n_stacks=n_stacks, layers=layers, **params)

    def call(self, sentences, training=False, return_vectors=False):
        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens

        # turn incoming sentences into relevant tensor batches
        with tf.name_scope('Encoding'):
            x_enc, enc_mask, _ = self.embed_layer(tokens)
            if not training:
                return self.encode_layer(x_enc, mask=enc_mask, training=False)
            x_enc, context, *states = self.encode_layer(x_enc, mask=enc_mask, training=True)

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
                training=training,
                initial_state=states
            )
            x_out = tf.tensordot(x_out, self.embed_layer.embeddings, axes=[2, 1])

        if return_vectors:
            return x_out, dec_time_steps, targets.to_tensor(default_value=0), context
        return x_out, dec_time_steps, targets.to_tensor(default_value=0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences):
        """Takes batches of free text and returns context vectors for each example.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        tf.Tensor
            Context vectors of shape (batch_size, embedding_size)
        """

        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        x_enc, enc_mask, _ = self.embed_layer(tokens)
        return self.encode_layer(x_enc, mask=enc_mask, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences):
        """Takes batches of free text and returns word embeddings along with the associate token.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            Tuple of (tokens, word_embeddings) with shapes (batch_size, max_sequence_len)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        return tokens.to_tensor(''), self.embed_layer(tokens, output_embeddings=True).to_tensor(0)


class ServingModel(tf.keras.Model):
    """Wrapper class for packaging final layers prior to saving.

        Parameters
        ----------
        embed_layer : TextInput
            Trained embedding layer.
        encode_layer : (TransformerEncoder or RecurrentEncoder)
            Trained encoding layer.
        sep : str, optional
            Token separator, by default ' '
        """

    def __init__(self, embed_layer, encode_layer, sep=' '):
        super().__init__()

        assert isinstance(embed_layer, TextInput)
        assert type(encode_layer) in {RecurrentEncoder, TransformerEncoder}

        self.embed_layer = embed_layer
        self.tokenizer = Tokenizer(sep)
        self.encode_layer = encode_layer

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def embed(self, sentences):
        """Takes batches of free text and returns context vectors for each example.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        tf.Tensor
            Context vectors of shape (batch_size, embedding_size)
        """

        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        x_enc, enc_mask, _ = self.embed_layer(tokens)
        return self.encode_layer(x_enc, mask=enc_mask, training=False)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def token_embed(self, sentences):
        """Takes batches of free text and returns word embeddings along with the associate token.

        Parameters
        ----------
        sentences : tf.Tensor
            Tensor of dtype tf.string.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            Tuple of (tokens, word_embeddings) with shapes (batch_size, max_sequence_len)
            and (batch_size, max_sequence_len, embedding_size) respectively.
        """

        tokens = self.tokenizer(sentences)  # turn sentences into ragged tensors of tokens
        return tokens.to_tensor(''), self.embed_layer(tokens, output_embeddings=True).to_tensor(0)


def sequence_cost(target_sequences, sequence_logits, num_labels, smoothing=False):
    """Sequence-to-sequence cost function with optional label smoothing.

    Parameters
    ----------
    target_sequences : tf.Tensor
        Expected token sequences as lookup IDs (batch_size, max_sequence_len)
    sequence_logits : [type]
        Computed logits for predicted tokens (batch_size, max_sequence_len, embedding_size)
    num_labels : int
        Vocabulary look up size
    smoothing : bool, optional
        Set to True to smooth labels, this increases regularization while increasing training time, by default False

    Returns
    -------
    tf.float32
        Loss value averaged over examples.
    """

    with tf.name_scope('Cost'):
        if smoothing:
            smoothing = 0.1
            targets = tf.one_hot(target_sequences, depth=num_labels, on_value=1.0, off_value=0.0, axis=-1)
            loss = tf.keras.losses.binary_crossentropy(
                y_true=targets,
                y_pred=sequence_logits,
                from_logits=True,
                label_smoothing=smoothing
            )
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sequence_logits, labels=target_sequences)

        loss = tf.reduce_mean(loss)
        return loss


def vector_cost(context_vectors):
    """Cost constraint on the cosine similarity of context vectors. Diagonal elements (self-context)
    are coerced to be closer to 1 (self-consistency). Off-diagonal elements are pushed toward 0, 
    indicating not contextually similar.

    Parameters
    ----------
    context_vectors : tf.Tensor
        (batch_size, embedding_size)

    Returns
    -------
    tf.float32
        cosine similarity constraint loss
    """

    with tf.name_scope('VectorCost'):
        rows = tf.shape(context_vectors)[0]
        context_vectors = tf.linalg.l2_normalize(context_vectors, axis=-1)
        cosine = tf.tensordot(context_vectors, tf.transpose(context_vectors), axes=[1, 0])
        identity = tf.eye(rows)
        return tf.reduce_mean((identity - cosine) ** 2)
