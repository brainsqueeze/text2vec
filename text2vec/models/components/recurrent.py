import tensorflow as tf


class BidirectionalLSTM(tf.keras.layers.Layer):

    def __init__(self, num_layers=2, num_hidden=32, return_states=False):
        super(BidirectionalLSTM, self).__init__()
        self.num_layers = num_layers
        self.return_states = return_states
        use_gpu = tf.test.is_gpu_available()
        lstm = tf.keras.layers.CuDNNLSTM if use_gpu else tf.keras.layers.LSTM

        params = dict(
            units=num_hidden,
            return_sequences=True,
            return_state=return_states,
            dtype=tf.float32
        )

        self.FWD = [lstm(**params, name="forward") for _ in range(num_layers)]
        self.BWD = [lstm(**params, name="backward", go_backwards=True) for _ in range(num_layers)]
        self.concat = tf.keras.layers.Concatenate()

    @staticmethod
    def __make_inputs(inputs, initial_states=None, layer=0):
        fwd_inputs = dict(inputs=inputs)
        bwd_inputs = dict(inputs=inputs)

        if initial_states is not None and layer == 0:
            fwd_inputs["initial_state"] = initial_states[0]
            bwd_inputs["initial_state"] = initial_states[1]

        return fwd_inputs, bwd_inputs

    def __call__(self, inputs, initial_states=None, training=False):
        layer = 0
        for forward, backward in zip(self.FWD, self.BWD):
            fwd_inputs, bwd_inputs = self.__make_inputs(inputs, initial_states=initial_states, layer=layer)

            if self.return_states:
                decode_forward, *forward_state = forward(**fwd_inputs)
                decode_backward, *backward_state = backward(**bwd_inputs)
            else:
                decode_forward = forward(**fwd_inputs)
                decode_backward = backward(**bwd_inputs)
            inputs = self.concat([decode_forward, decode_backward])
            layer += 1
        if self.return_states:
            return inputs, [forward_state, backward_state]
        return inputs
