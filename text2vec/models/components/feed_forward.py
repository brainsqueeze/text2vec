import tensorflow as tf


class PositionWiseFFN(object):

    def __init__(self, emb_dims):
        hidden_dim_size = 4 * emb_dims

        self.ConvInner = tf.Variable(
            tf.zeros([1, emb_dims, hidden_dim_size]),
            name='conv-filter-inner',
            dtype=tf.float32,
            trainable=True
        )
        self.ConvOuter = tf.Variable(
            tf.zeros([1, hidden_dim_size, emb_dims]),
            name='conv-filter-outer',
            dtype=tf.float32,
            trainable=True
        )

    def __call__(self, x):
        with tf.name_scope("PositionWiseFeedForward"):
            x = tf.nn.conv1d(x, filters=self.ConvInner, stride=1, padding='SAME')
            x = tf.nn.relu(x)
            return tf.nn.conv1d(x, filters=self.ConvOuter, stride=1, padding='SAME')
