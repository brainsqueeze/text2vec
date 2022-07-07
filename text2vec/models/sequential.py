import tensorflow as tf
from tensorflow.keras import layers

from .components.attention import BahdanauAttention
from .components.recurrent import BidirectionalLSTM
from .components.utils import TensorProjection


class RecurrentEncoder(layers.Layer):
    """LSTM based encoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    num_hidden : int
        Dimensionality of hidden LSTM layer weights.
    num_layers : int, optional
        Number of hidden LSTM layers, by default 2
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import TextInputs
    from text2vec.models import RecurrentEncoder

    lookup = {'string': 0, 'is': 1, 'example': 2}
    inputer = TextInput(token_hash=lookup, embedding_size=16, max_sequence_len=10)
    encoder = RecurrentEncoder(max_sequence_len=10, num_hidden=8, input_keep_prob=0.75)

    text = tf.ragged.constant([
        ["Sample", "string", "."],
        ["This", "is", "a", "second", "example", "."]
    ])
    x, mask, _ = inputer(text)
    x_enc, context, *states = encoder(x_enc, mask=enc_mask, training=True)
    ```
    """

    def __init__(self, max_sequence_len, num_hidden, num_layers=2,
                 input_drop_rate: float = 0., hidden_drop_rate: float = 0., **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_len

        self.drop = layers.Dropout(input_drop_rate)
        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=True)
        self.attention = BahdanauAttention(size=2 * num_hidden, drop_rate=hidden_drop_rate)

    # pylint: disable=missing-function-docstring
    def call(self, x, mask, training: bool = False):
        with tf.name_scope("RecurrentEncoder"):
            mask = tf.expand_dims(mask, axis=-1)
            x = self.drop(x, training=training)
            x, states = self.bi_lstm(x * mask, training=training)
            x, context = self.attention(x * mask)

            if training:
                return x, context, states
            return x, context


class RecurrentDecoder(layers.Layer):
    """LSTM based decoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    num_hidden : int
        Dimensionality of hidden LSTM layer weights.
    embedding_size : int, optional
        Dimensionality of the word-embeddings, by default 50.
    num_layers : int, optional
        Number of hidden LSTM layers, by default 2
    input_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    hidden_drop_rate : float, optional
        Value between 0 and 1.0, by default 0.
    """

    def __init__(self, max_sequence_len, num_hidden, embedding_size=50, num_layers=2,
                 input_drop_rate: float = 0., hidden_drop_rate: float = 0.):
        super().__init__()
        self.max_sequence_length = max_sequence_len
        dims = embedding_size

        self.drop = layers.Dropout(input_drop_rate)
        self.h_drop = layers.Dropout(hidden_drop_rate)
        self.projection = TensorProjection()

        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=False)
        self.dense = layers.Dense(units=dims, activation=tf.nn.relu)

    # pylint: disable=missing-function-docstring
    def call(self, x_enc, x_dec, dec_mask, context, initial_state=None, training: bool = False):
        dec_mask = tf.expand_dims(dec_mask, axis=-1)

        x = self.drop(x_dec * dec_mask, training=training)
        if initial_state is not None:
            x = self.bi_lstm(x * dec_mask, initial_states=initial_state[0], training=training)
        else:
            x = self.bi_lstm(x * dec_mask, training=training)
        x = self.h_drop(self.projection(x, projection_vector=context), training=training)
        x = self.dense(x * dec_mask)
        return x
