from text2vec.models import InputFeeder
from text2vec.models import BahdanauAttention
from text2vec.models import MultiHeadAttention
from text2vec.models import PositionWiseFFN
from text2vec.models import utils

import tensorflow as tf
from text2vec.preprocessing import utils as str_utils

from functools import partial

tf.enable_eager_execution()
SIZE = 1000
DIMS = 64

test_sentences = [
    "The movie was great!",
    "The movie was terrible."
]
test_tokens = [' '.join(str_utils.clean_and_split(text)) for text in test_sentences]
hash_map, max_sequence_length = str_utils.get_top_tokens(test_sentences, n_top=SIZE)

token_feed = InputFeeder(token_hash=hash_map, num_labels=SIZE, emb_dims=DIMS)
multi_head_attention = MultiHeadAttention(emb_dims=DIMS)
ffn = PositionWiseFFN(emb_dims=DIMS)
attention = BahdanauAttention(size=DIMS)
positional_encoder = utils.positional_encode(emb_dims=DIMS, max_sequence_length=max_sequence_length)


@tf.function
def encode(sentences, input_keep_prob=1.0, hidden_keep_prob=1.0, stacks=1):
    tokens = tf.string_split(sentences, sep=' ')
    tokens = tf.RaggedTensor.from_sparse(tokens)

    x, enc_mask, _ = token_feed(tokens, max_sequence_length=max_sequence_length)
    x = x + (positional_encoder * enc_mask)
    x = tf.nn.dropout(x, rate=1 - input_keep_prob)

    h_dropout = partial(tf.nn.dropout, rate=1 - hidden_keep_prob)

    for _ in range(stacks):
        x = h_dropout(multi_head_attention(*([x] * 3))) + x
        x = utils.layer_norm_compute(x)
        x = h_dropout(ffn(x)) + x
        x = utils.layer_norm_compute(x)

    return attention(x * enc_mask)


if __name__ == '__main__':
    output = encode(test_tokens)
    print(output)
    # print(output.numpy())
