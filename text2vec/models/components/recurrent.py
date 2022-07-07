import tensorflow as tf
from tensorflow.keras import layers


class BidirectionalLSTM(layers.Layer):
    """Bi-directional LSTM with the option to warm initialize with previous states.

    Parameters
    ----------
    num_layers : int, optional
        Number of hidden LSTM layers, by default 2
    num_hidden : int, optional
        Dimensionality of hidden LSTM layer weights, by default 32
    return_states : bool, optional
        Flag to set whether the internal LSTM states should be returned. This is useful for
        warm initializations, by default False

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models import BidirectionalLSTM

    encoded_sequences = tf.random.uniform(shape=[4, 7, 12])
    decoded_sequences = tf.random.uniform(shape=[4, 11, 12])

    encode = BidirectionalLSTM(num_layers=2, num_hidden=16, return_states=True)
    decode = BidirectionalLSTM(num_layers=2, num_hidden=16)
    x, states = encode(encoded_sequences)
    y = decode(decoded_sequences, initial_states=states)
    ```
    """

    def __init__(self, num_layers: int = 2, num_hidden: int = 32, return_states: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.return_states = return_states
        lstm = layers.LSTM

        params = dict(
            units=num_hidden,
            return_sequences=True,
            return_state=return_states,
            dtype=tf.float32
        )

        self.FWD = [lstm(**params, name=f"forward-{i}") for i in range(num_layers)]
        self.BWD = [lstm(**params, name=f"backward-{i}", go_backwards=True) for i in range(num_layers)]
        self.concat = layers.Concatenate()

    @staticmethod
    def __make_inputs(inputs, initial_states=None, layer=0):
        fwd_inputs = dict(inputs=inputs)
        bwd_inputs = dict(inputs=inputs)

        if initial_states is not None and layer == 0:
            fwd_inputs["initial_state"] = initial_states[0]
            bwd_inputs["initial_state"] = initial_states[1]

        return fwd_inputs, bwd_inputs

    def call(self, inputs, initial_states=None, training=False):
        layer = 0
        for forward, backward in zip(self.FWD, self.BWD):
            fwd_inputs, bwd_inputs = self.__make_inputs(inputs, initial_states=initial_states, layer=layer)

            if self.return_states:
                decode_forward, *forward_state = forward(**fwd_inputs, training=training)
                decode_backward, *backward_state = backward(**bwd_inputs, training=training)
            else:
                decode_forward = forward(**fwd_inputs, training=training)
                decode_backward = backward(**bwd_inputs, training=training)
            inputs = self.concat([decode_forward, decode_backward])
            layer += 1
        if self.return_states:
            return inputs, [forward_state, backward_state]
        return inputs
