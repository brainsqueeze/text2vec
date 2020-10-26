from collections import Counter
import tensorflow as tf


def get_top_tokens(corpus, n_top=1000):
    """
    Builds the token mapping which is used to initialize the word embeddings in the model.
    Get the most frequent terms which appear in the training corpus.

    Parameters
    ----------
    corpus : tf.data.Dataset
        Entire dataset object
    n_top : int, optional
        Number of most frequenct vocab terms to keep for training, by default 1000

    Returns
    -------
    (dict, int, int)
        (token->integer lookup, maximum sequence length, size of data set)
    """

    assert isinstance(corpus, tf.data.Dataset)

    lookup = Counter()
    max_sequence_length, data_set_size = 0, 0

    corpus = corpus.map(lambda x: tf.strings.split(x, sep=''), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for tokens_list in corpus.apply(tf.data.experimental.dense_to_ragged_batch(32)).prefetch(5):
        lookup.update(tokens_list.flat_values.numpy())

        max_batch_seq_len = int(tokens_list.row_lengths().numpy().max())
        if max_batch_seq_len > max_sequence_length:
            max_sequence_length = max_batch_seq_len
        data_set_size += int(tokens_list.nrows())

    # tensorflow converts strings to bytes, let's maintain that (no decoding)
    hash_map = {key: idx + 2 for idx, (key, value) in enumerate(lookup.most_common(n_top))}
    hash_map["<s>".encode('utf8')] = 0
    hash_map["</s>".encode('utf8')] = 1
    return hash_map, max_sequence_length, data_set_size
