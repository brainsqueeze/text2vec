import tensorflow as tf

from text2vec.models import Tokenizer


class SubStringFinderMask(tf.keras.layers.Layer):
    """Performs substring masking based on whether the substring is found in the input text
    in its entirety. This returns a ragged boolean tensor with the same ragged shape as input substrings.

    Parameters
    ----------
    sep : str, optional
        The token to split the incoming strings by, by default ' '

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models.components.strings import SubStringFinderMask

    text = tf.constant([
        "Sample string.",
        "This is a second example."
    ])

    substrings = tf.ragged.constant([
        ["string"],
        ["this", "apple", "example"]
    ])
    mask_agent = SubStringFinderMask()
    mask_agent(text, substrings)
    ```
    """

    def __init__(self, sep: str = ' '):
        super().__init__()
        self.tokenizer = Tokenizer(sep)
        self.match = tf.keras.layers.Lambda(lambda x: tf.strings.regex_full_match(input=x[0], pattern=x[1]))

    def find_match(self, texts, substrings):
        return tf.map_fn(self.match, tf.stack([texts, substrings], axis=1), fn_output_signature=tf.bool)

    def call(self, texts: tf.Tensor, substrings: tf.RaggedTensor) -> tf.RaggedTensor:
        pre = '.*(\s|^)'
        post = '(\s|$).*'

        ragged_texts = tf.RaggedTensor.from_row_lengths(
            values=tf.repeat(texts, repeats=substrings.row_lengths()),
            row_lengths=substrings.row_lengths()
        )
        return tf.ragged.map_flat_values(self.find_match, ragged_texts, tf.strings.join([pre, substrings, post]))
