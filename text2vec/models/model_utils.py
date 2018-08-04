import tensorflow as tf


def build_cell(num_layers, num_hidden, keep_prob, use_cuda=False):
    cells = []

    for _ in range(num_layers):
        if use_cuda:
            cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)
        else:
            cell = tf.nn.rnn_cell.LSTMCell(
                num_hidden,
                forget_bias=0.0,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
            )

        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        cells.append(cell)

    if num_layers > 1:
        return tf.nn.rnn_cell.MultiRNNCell(cells)
    return cells[0]


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    _length = tf.reduce_sum(used, 1)
    _length = tf.cast(_length, tf.int32)
    return _length


def sum_reduce(seq_fw, seq_bw):
    if tf.contrib.framework.nest.is_sequence(seq_fw):
        tf.contrib.framework.nest.assert_same_structure(seq_fw, seq_bw)

        x_flat = tf.contrib.framework.nest.flatten(seq_fw)
        y_flat = tf.contrib.framework.nest.flatten(seq_bw)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(tf.add_n([x_i, y_i]))

        return tf.contrib.framework.nest.pack_sequence_as(seq_fw, flat)
    return tf.add_n([seq_fw, seq_bw])


def concat_reducer(seq_fw, seq_bw):
    if tf.contrib.framework.nest.is_sequence(seq_fw):
        tf.contrib.framework.nest.assert_same_structure(seq_fw, seq_bw)

        x_flat = tf.contrib.framework.nest.flatten(seq_fw)
        y_flat = tf.contrib.framework.nest.flatten(seq_bw)

        flat = []
        for x_i, y_i in zip(x_flat, y_flat):
            flat.append(tf.concat([x_i, y_i], axis=-1))

        return tf.contrib.framework.nest.pack_sequence_as(seq_fw, flat)
    return tf.concat([seq_fw, seq_bw], axis=-1)
