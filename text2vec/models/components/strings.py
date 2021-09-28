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

        # this is designed to approximate the functionality in re.escape
        self.special_chars = r'[\(\)\[\]\{\}\?\*\+\-\|\^\$\\\\\.\&\~\#\\\t\\\n\\\r\\\v\\\f]'

    def find_match(self, texts: tf.Tensor, substrings: tf.Tensor) -> tf.Tensor:
        """Vectorized regex matching with row-aligned texts and substrings.

        Parameters
        ----------
        texts : tf.Tensor
            Tensor of strings to search on
        substrings : tf.Tensor
            Tensor of substrings to search for, must be aligned row-wise with the texts

        Returns
        -------
        tf.Tensor
            Boolean tensor with the same shape as `texts` and `substrings`
        """

        return tf.map_fn(self.match, tf.stack([texts, substrings], axis=1), fn_output_signature=tf.bool)

    def call(self, texts: tf.Tensor, substrings: tf.RaggedTensor) -> tf.RaggedTensor:
        texts = tf.strings.regex_replace(texts, pattern=self.special_chars, rewrite='')
        texts = tf.strings.strip(tf.strings.regex_replace(texts, pattern=r'\s{2,}', rewrite=' '))

        substrings = tf.strings.regex_replace(substrings, pattern=self.special_chars, rewrite='')
        substrings = tf.strings.strip(tf.strings.regex_replace(substrings, pattern=r'\s{2,}', rewrite=' '))

        pre = r'.*(\s|^)'
        post = r'(\s|$).*'

        ragged_texts = tf.RaggedTensor.from_row_lengths(
            values=tf.repeat(texts, repeats=substrings.row_lengths()),
            row_lengths=substrings.row_lengths()
        )
        return tf.ragged.map_flat_values(self.find_match, ragged_texts, tf.strings.join([pre, substrings, post]))
