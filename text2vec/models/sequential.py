import tensorflow as tf

from .components.attention import BahdanauAttention
from .components.recurrent import BidirectionalLSTM
from .components.utils import TensorProjection


class RecurrentEncoder(tf.keras.layers.Layer):
    """LSTM based encoding pipeline.

    Parameters
    ----------
    max_sequence_len : int
        Longest sequence seen at training time.
    num_hidden : int
        Dimensionality of hidden LSTM layer weights.
    num_layers : int, optional
        Number of hidden LSTM layers, by default 2
    input_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.

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

    def __init__(self, max_sequence_len, num_hidden, num_layers=2, input_keep_prob=1.0, **kwargs):
        super().__init__()
        self.max_sequence_length = max_sequence_len

        self.drop = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=True)
        self.attention = BahdanauAttention(size=2 * num_hidden)

    def call(self, x, mask, training=False, **kwargs):
        with tf.name_scope("RecurrentEncoder"):
            x = self.drop(x, training=training)
            x, states = self.bi_lstm(x)
            context = self.attention(x)

            if training:
                return x, context, states
            return x, context


class RecurrentDecoder(tf.keras.layers.Layer):
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
    input_keep_prob : float, optional
        Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    hidden_keep_prob : float, optional
        Hidden states dropout. Value between 0 and 1.0 which determines `1 - dropout_rate`, by default 1.0.
    """

    def __init__(self, max_sequence_len, num_hidden, embedding_size=50, num_layers=2,
                 input_keep_prob=1.0, hidden_keep_prob=1.0):
        super().__init__()
        self.max_sequence_length = max_sequence_len
        dims = embedding_size

        self.drop = tf.keras.layers.Dropout(1 - input_keep_prob, name="InputDropout")
        self.h_drop = tf.keras.layers.Dropout(1 - hidden_keep_prob, name="HiddenStateDropout")
        self.projection = TensorProjection()

        self.bi_lstm = BidirectionalLSTM(num_layers=num_layers, num_hidden=num_hidden, return_states=False)
        self.dense = tf.keras.layers.Dense(units=dims, activation=tf.nn.relu)

    def call(self, x_enc, enc_mask, x_dec, dec_mask, context, attention, training=False, **kwargs):
        with tf.name_scope("RecurrentDecoder"):
            initial_state = kwargs.get("initial_state")
            x = self.drop(x_dec, training=training)
            if initial_state is not None:
                x = self.bi_lstm(x, initial_states=initial_state[0])
            else:
                x = self.bi_lstm(x)
            x = self.h_drop(self.projection(x, projection_vector=context))
            x = self.dense(x)
            return x
