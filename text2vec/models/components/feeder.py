import tensorflow as tf
from .utils import ragged_tensor_process_mask


class InputFeeder(object):

    def __init__(self, token_hash, emb_dims):
        assert isinstance(token_hash, dict)

        self.num_labels = len(token_hash)
        self.table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(list(token_hash.keys()), list(token_hash.values())),
            default_value=max(token_hash.values()) + 1
        )
        self.embeddings = tf.Variable(
            tf.random.uniform([self.num_labels, emb_dims], -1.0, 1.0),
            name='embeddings',
            dtype=tf.float32,
            trainable=True
        )

    def token_lookup(self, x):
        assert isinstance(x, tf.RaggedTensor)
        return tf.ragged.map_flat_values(self.table.lookup, x)

    def __call__(self, x, max_sequence_length):
        assert isinstance(x, tf.RaggedTensor)
        output, mask, time_steps = ragged_tensor_process_mask(
            tokens=x,
            lookup=self.table,
            embeddings=self.embeddings,
            max_sequence_length=max_sequence_length
        )
        return output, mask, time_steps
