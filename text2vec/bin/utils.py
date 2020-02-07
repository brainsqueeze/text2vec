import numpy as np


def log(message):
    print(f"[INFO] {message}")


def test_val_split(corpus, val_size):
    """
    Splits the entire corpus into training and validation sets
    :param corpus: all training documents (list)
    :param val_size: number of examples in the validation set (int)
    :return: training set, validation set (list, list)
    """

    s = np.random.permutation(range(len(corpus)))
    cv_set = [corpus[item] for item in s[:val_size]]
    corpus = [corpus[item] for item in s[val_size:]]
    return corpus, cv_set


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
