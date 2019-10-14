from text2vec import training_tools

from text2vec.models import InputFeeder
from text2vec.models import TransformerEncoder
from text2vec.models import TransformerDecoder

import tensorflow as tf
from text2vec.preprocessing import utils as str_utils

tf.enable_eager_execution()
SIZE = 1000
DIMS = 64

test_sentences = [
    "The movie was great!",
    "The movie was terrible."
]
test_tokens = [' '.join(str_utils.clean_and_split(text)) for text in test_sentences]
hash_map, max_sequence_length = str_utils.get_top_tokens(test_sentences, n_top=SIZE)

token_feed = InputFeeder(token_hash=hash_map, emb_dims=DIMS)
encoder = TransformerEncoder(max_sequence_len=max_sequence_length, embedding_size=DIMS)
decoder = TransformerDecoder(max_sequence_len=max_sequence_length, num_labels=SIZE, embedding_size=DIMS)


if __name__ == '__main__':
    output = training_tools.train_step(test_tokens, inputs_handler=token_feed, encoder=encoder, decoder=decoder)
    print(output)
