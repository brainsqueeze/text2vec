import numpy as np


def log(message, **kwargs):
    print(f"[INFO] {message}", flush=True, end=kwargs.get("end", "\n"))


def compute_angles(vectors):
    """Computes the angles between vectors

    Parameters
    ----------
    vectors : np.ndarray
        (batch_size, embedding_size)

    Returns
    -------
    np.ndarray
        Cosine angles in degrees (batch_size, batch_size)
    """

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.dot(vectors, vectors.T)
    cosine = np.clip(cosine, -1, 1)
    degrees = np.arccos(cosine) * (180 / np.pi)
    return degrees
