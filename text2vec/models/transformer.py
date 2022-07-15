from tensorflow.keras import layers

from .components.attention import BahdanauAttention, MultiHeadAttention
from .components.feed_forward import PositionWiseFFN
from .components.utils import VariationPositionalEncoder, LayerNorm, TensorProjection


class TransformerEncoder(layers.Layer):
    """Attention based encoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    num_layers : int, optional
        Number of layers in the multi-head-attention layer, by default 8
    n_stacks : int, optional
        Number of encoding blocks to chain, by default 1
    embedding_size : int, optional
        Dimensionality of the word-embeddings, by default 50.
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    hidden_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TokenEmbed
    from text2vec.models import TransformerEncoder

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TokenEmbed(token_hash=lookup, embedding_size=16, max_sequence_len=10)
    encoder = TransformerEncoder(max_sequence_len=10, embedding_size=16, input_keep_prob=0.75)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    x, mask, _ = inputer(text)
    x, context = encoder(x_enc, mask=enc_mask, training=True)
    ```
    """

    def __init__(self, max_sequence_len, num_layers=8, n_stacks=1, embedding_size=50,
                 input_drop_rate: float = 0., hidden_drop_rate: float = 0.):
        super().__init__()
        dims = embedding_size

        self.positional_encode = VariationPositionalEncoder(emb_dims=dims, max_sequence_len=max_sequence_len)
        self.layer_norm = LayerNorm()
        self.MHA = [
            MultiHeadAttention(emb_dims=dims, num_layers=num_layers, drop_rate=input_drop_rate)
            for _ in range(n_stacks)
        ]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]
        self.attention = BahdanauAttention(size=dims, drop_rate=hidden_drop_rate)

        self.drop = layers.Dropout(input_drop_rate)
        self.h_drop = layers.Dropout(hidden_drop_rate)

    # pylint: disable=missing-function-docstring
    def call(self, x, mask, training: bool = False):
        x = self.positional_encode(x, mask)
        x = self.drop(x, training=training)

        for mha, ffn in zip(self.MHA, self.FFN):
            x = self.h_drop(mha([x] * 3, training=training), training=training) + x
            x = self.layer_norm(x)
            x = self.h_drop(ffn(x), training=training) + x
            x = self.layer_norm(x)

        x, context = self.attention(x, training=training)
        return x, context


class TransformerDecoder(layers.Layer):
    """Attention based decoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    num_layers : int, optional
        Number of layers in the multi-head-attention layer, by default 8
    n_stacks : int, optional
        Number of encoding blocks to chain, by default 1
    embedding_size : int, optional
        Dimensionality of the word-embeddings, by default 50.
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    hidden_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    """

    def __init__(self, max_sequence_len, num_layers=8, n_stacks=1, embedding_size=50,
                 input_drop_rate: float = 0., hidden_drop_rate: float = 0.):
        super().__init__()
        dims = embedding_size

        self.layer_norm = LayerNorm()
        self.projection = TensorProjection()
        self.positional_encode = VariationPositionalEncoder(emb_dims=dims, max_sequence_len=max_sequence_len)
        self.MHA = [
            MultiHeadAttention(emb_dims=dims, num_layers=num_layers, drop_rate=input_drop_rate)
            for _ in range(n_stacks)
        ]
        self.FFN = [PositionWiseFFN(emb_dims=dims) for _ in range(n_stacks)]

        self.drop = layers.Dropout(input_drop_rate)
        self.h_drop = layers.Dropout(hidden_drop_rate)

    # pylint: disable=missing-function-docstring
    def call(self, x_enc, x_dec, dec_mask, context, attention: BahdanauAttention, training: bool = False):
        x_dec = self.positional_encode(x_dec, dec_mask)
        x_dec = self.drop(x_dec, training=training)

        for mha, ffn in zip(self.MHA, self.FFN):
            x_dec = self.h_drop(mha(
                [x_dec] * 3,
                mask_future=True,
                training=training
            ), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)

            x_dec, cross_context = attention(encoded=x_enc, decoded=x_dec, training=training)
            x_dec = self.h_drop(self.projection(x_dec, projection_vector=cross_context), training=training) + x_dec

            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_drop(ffn(x_dec), training=training) + x_dec
            x_dec = self.layer_norm(x_dec)
            x_dec = self.h_drop(self.projection(x_dec, projection_vector=context), training=training) + x_dec
        return x_dec
