import tensorflow as tf
from tensorflow.keras import layers

from .components.attention import BahdanauAttention
from .components.attention import MultiHeadAttention
from .components.feed_forward import PositionWiseFFN
from .components.utils import PositionalEncoder
from .components.utils import LayerNorm
from .components.utils import TensorProjection


class TransformerEncoder(layers.Layer):
    """Attention based encoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    layers : int, optional
        Number of layers in the multi-head-attention layer, by default 8
    n_stacks : int, optional
        Number of encoding blocks to chain, by default 1
    embedding_size : int, optional
        Dimensionality of the word-embeddings, by default 50.
    input_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    hidden_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TextInputs
    from text2vec.models import TransformerEncoder

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TextInput(token_hash=lookup, embedding_size=16, max_sequence_len=10)
    encoder = TransformerEncoder(max_sequence_len=10, embedding_size=16, input_keep_prob=0.75)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    x, mask, _ = inputer(text)
    x, context = encoder(x_enc, mask=enc_mask, training=True)
    ```
    """

    def __init__(self, max_sequence_len, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super().__init__()
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.drop = layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_drop = layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.layer_norm = LayerNorm()

        self.positional_encode = PositionalEncoder(emb_dims=dims, max_sequence_len=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.attention = BahdanauAttention(size=dims)

    def call(self, x, mask, training=False):
        x = self.positional_encode(x, mask)
        x = self.drop(x, training=training)

        for mha, ffn in zip(self.MHA, self.FFN):
            x = self.h_drop(mha([x] * 3, training=training), training=training) + x
            x = self.layer_norm(x)
            x = self.h_drop(ffn(x), training=training) + x
            x = self.layer_norm(x)

        context = self.attention(x)
        return x, context


class TransformerDecoder(layers.Layer):
    """Attention based decoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    layers : int, optional
        Number of layers in the multi-head-attention layer, by default 8
    n_stacks : int, optional
        Number of encoding blocks to chain, by default 1
    embedding_size : int, optional
        Dimensionality of the word-embeddings, by default 50.
    input_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    hidden_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    """

    def __init__(self, max_sequence_len, layers=8, n_stacks=1, embedding_size=50,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super().__init__()
        dims = embedding_size
        keep_prob = hidden_keep_prob

        self.drop = layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_drop = layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.layer_norm = LayerNorm()
        self.projection = TensorProjection()

        self.positional_encode = PositionalEncoder(emb_dims=dims, max_sequence_len=max_sequence_len)
        self.MHA = [MultiHeadAttention(emb_dims=dims, layers=layers, keep_prob=keep_prob) for _ in range(n_stacks)]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]

    def call(self, x_enc, enc_mask, x_dec, dec_mask, context, attention, training=False, **kwargs):
        x_dec = self.positional_encode(x_dec, dec_mask)
        x_dec = self.drop(x_dec, training=training)

        for mha, ffn in zip(self.MHA, self.FFN):
            x_dec = self.h_drop(mha(
                [x_dec] * 3,
                mask_future=True,
                training=training
            ), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)

            cross_context = attention(encoded=x_enc, decoded=x_dec)
            x_dec = self.h_drop(self.projection(x_dec, projection_vector=cross_context), training=training) + x_dec

            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_drop(ffn(x_dec), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_drop(self.projection(x_dec, projection_vector=context), training=training) + x_dec
        return x_dec
