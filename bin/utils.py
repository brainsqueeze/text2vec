import numpy as np


def log(message):
    print(f"[INFO] {message}")


def load_glove_vectors(lookup, glove_path):
    """
    Loads pre-trained GloVe embeddings from file
    :param lookup: embedding lookup object (EmbeddingLookup instance)
    :param glove_path: full file path to the GloVe embeddings (str)
    :return: weights, vocab (np.ndarray, set)
    """

    weights, ordering, vocab = [], [], set()
    count = 0
    with open(glove_path, 'r') as f:
        for line in f:
            split = line.split()
            size = len(split)
            values = split[-300:]
            token = ' '.join(split[:size - 300])

            if token in lookup:
                vocab.add(token)
                model_order = lookup[token]
                weights.append(list(map(float, values)))
                ordering.append((count, model_order))
                count += 1

            if count > len(lookup):
                break

    ordering, _ = zip(*sorted(ordering, key=lambda idx: idx[1]))
    weights = np.array(weights, dtype=np.float32)
    weights = weights[np.array(ordering)]
    unknown = weights.mean(axis=0)[:, None].T
    pad_vector = np.random.random((1, weights.shape[1]))
    weights = np.vstack([pad_vector, weights, unknown])
    return weights, vocab


def pad_sequence(sequence, max_sequence_length):
    """
    Pads individual text sequences to the maximum length
    seen by the model at training time
    :param sequence: list of integer lookup keys for the vocabulary (list)
    :param max_sequence_length: (int)
    :return: padded sequence (ndarray)
    """

    sequence = np.array(sequence, dtype=np.int32)
    difference = max_sequence_length - sequence.shape[0]
    pad = np.zeros((difference,), dtype=np.int32)
    return np.concatenate((sequence, pad))


def compute_angles(vectors):
    """
    Computes the angles between vectors
    :param vectors: (np.ndarray)
    :return: angles in degrees, output is [num_examples, vectors_dimension] (np.ndarray)
    """

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.dot(vectors, vectors.T)
    cosine = np.clip(cosine, -1, 1)
    degrees = np.arccos(cosine) * (180 / np.pi)
    return degrees
