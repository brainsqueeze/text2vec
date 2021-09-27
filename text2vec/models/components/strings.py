import tensorflow as tf

from text2vec.models import Tokenizer


class SubTokenFinderMask(tf.keras.layers.Layer):
    """Performs substring masking based on whether the substring is found in the input text
    either in part or in entirety. The substring search implementation is simple and is performed by tokenizing
    the input text with some specified separator. If you require more complex searching, then it is advised that
    the input text be pre-processed with a more robust tokenizer, such as BPE or WordPiece.

    Parameters
    ----------
    sep : str, optional
        The token to split the incoming strings by, by default ' '
    mode : str, optional
        The final reduction mode. Setting `mode` to 'all' will mask the entire substring if any subtokens are not found,
        by default 'any'

    Examples
    --------
    ```python
    import tensorflow as tf
    from text2vec.models.components.strings import SubTokenFinderMask

    text = tf.constant([
        "Sample string.",
        "This is a second example."
    ])

    substrings = tf.ragged.constant([
        ["string"],
        ["this", "apple", "example"]
    ])
    mask_agent = SubTokenFinderMask()
    mask_agent(text, substrings)
    ```
    """

    def __init__(self, sep: str = ' ', mode: str = 'any'):
        super().__init__()

        self.tokenizer = Tokenizer(sep)
        if mode == 'all':
            self.reducer = tf.math.reduce_all
        else:
            self.reducer = tf.math.reduce_any

    @staticmethod
    def find_token(text_tokens: tf.Tensor, token) -> tf.Tensor:
        """Finds token in the input text as a Tensor element search

        Parameters
        ----------
        text_tokens : tf.Tensor
            String to search on, split into sub-tokens
        token : String-like scalar
            Token to find

        Returns
        -------
        tf.Tensor
            Boolean scalar tensor
        """

        return tf.math.reduce_any(text_tokens == token)

    def find_token_sequence(self, text_tokens: tf.Tensor, token_sequence: tf.Tensor) -> tf.Tensor:
        """Finds sequence of tokens in the input text as a Tensor element search

        Parameters
        ----------
        text_tokens : tf.Tensor
            String to search on, split into sub-tokens
        token_sequence : tf.Tensor
            Sequence of tokens to find

        Returns
        -------
        tf.Tensor
            Boolean vector tensor
        """

        return tf.map_fn(lambda x: self.find_token(text_tokens, x), token_sequence, fn_output_signature=tf.bool)

    def find_token_sequence_set(self, text_tokens: tf.Tensor, token_sequence_set: tf.RaggedTensor) -> tf.RaggedTensor:
        """Finds set of token sequences in the input text

        Parameters
        ----------
        text_tokens : tf.Tensor
            String to search on, split into sub-tokens
        token_sequence_set : tf.RaggedTensor
            Set of token sequences

        Returns
        -------
        tf.RaggedTensor
            Ragged boolean tensor
        """

        return tf.ragged.map_flat_values(lambda x: self.find_token_sequence(text_tokens, x), token_sequence_set)

    def call(self, text: tf.Tensor, substrings: tf.RaggedTensor, token_level_out: bool = True) -> tf.RaggedTensor:
        combined = tf.concat([
            tf.expand_dims(self.tokenizer(text), axis=1),
            self.tokenizer(substrings)
        ], axis=1)

        token_found = tf.map_fn(
            lambda x: self.find_token_sequence_set(x[0], x[1:]),
            combined,
            fn_output_signature=tf.RaggedTensorSpec(ragged_rank=1, dtype=tf.bool)
        )
        
        if token_level_out:
            return token_found
        return self.reducer(token_found, axis=-1)
