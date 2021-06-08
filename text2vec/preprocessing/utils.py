from collections import Counter
from typing import List, Tuple, Union
from glob import glob
import os

import tensorflow as tf


def get_top_tokens(corpus: tf.data.Dataset, n_top: int = 1000) -> Tuple[dict, int, int]:
    """
    Builds the token mapping which is used to initialize the word embeddings in the model.
    Get the most frequent terms which appear in the training corpus.

    Parameters
    ----------
    corpus : tf.data.Dataset
        Entire dataset object
    n_top : int, optional
        Number of most frequent vocab terms to keep for training, by default 1000

    Returns
    -------
    (dict, int, int)
        (token->integer lookup, maximum sequence length, size of data set)
    """

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


def check_valid(text: tf.string, max_length: int) -> bool:
    """Validates a sentence string for inclusion in the training set

    Parameters
    ----------
    text : tf.string
        Input string tensor
    max_length : int
        Maximum sequence length

    Returns
    -------
    bool
        True if string is not empty and has a sequence shorter than the defined maximum length.
    """

    sequence_lengths = tf.shape(tf.strings.split(text, sep=''))
    if max_length < 0:
        return sequence_lengths is not None
    return sequence_lengths is not None and sequence_lengths[0] <= max_length


def load_text_files(data_files: Union[List[str], str], max_length: int) -> tf.data.Dataset:
    """Loads the training data from a text file.

    Parameters
    ----------
    data_files : Union[List[str], str]
        Path string, or list of path strings to training data set files. Paths must be absolute.
    max_length : int
        Maximum sequence length to allow.

    Returns
    -------
    tf.data.Dataset
        Text data set.

    Raises
    ------
    ValueError
        If no data is specified.

    ValueError
        If no valid data could be discovered.
    """

    files = []
    if isinstance(data_files, list) and len(data_files) > 0:
        for f in data_files:
            if '*' in f:
                files.extend(glob(f))
                continue
            if os.path.isfile(f):
                files.append(f)
    elif isinstance(data_files, str):
        if '*' in data_files:
            files.extend(glob(data_files))
        if os.path.isfile(data_files):
            files.append(data_files)
    else:
        raise ValueError("Please specify one or more data files.")

    if len(files) == 0:
        raise ValueError("No valid files could be found.")

    texts = tf.data.TextLineDataset(files)
    texts = texts.map(tf.strings.strip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return texts.filter(lambda x: check_valid(x, max_length))
