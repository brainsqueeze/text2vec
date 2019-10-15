from text2vec import training_tools
from text2vec.optimizer_tools import RampUpDecaySchedule

from text2vec.models import InputFeeder
from text2vec.models import TransformerEncoder
from text2vec.models import TransformerDecoder

import tensorflow as tf
from text2vec.preprocessing import utils as text_utils

tf.enable_eager_execution()
# tf.disable_eager_execution()
SIZE = 1000
DIMS = 64

test_sentences = [
    "The movie was great!",
    "The movie was terrible."
]
test_tokens = [' '.join(text_utils.clean_and_split(text)) for text in test_sentences]
hash_map, max_sequence_length = text_utils.get_top_tokens(test_sentences, n_top=SIZE)

token_feed = InputFeeder(token_hash=hash_map, emb_dims=DIMS)
encoder = TransformerEncoder(max_sequence_len=max_sequence_length, embedding_size=DIMS)
decoder = TransformerDecoder(max_sequence_len=max_sequence_length, num_labels=len(hash_map), embedding_size=DIMS)
learning_rate = RampUpDecaySchedule(DIMS)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


if __name__ == '__main__':
    # checkpoint_path = './checkpoints/train'
    # ckpt = tf.train.Checkpoint(encoder=encoder, optimizer=optimizer)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # train_loss = tf.keras.metrics.Mean(name="train_loss")
    #
    # train_loss.reset_states()

    for step in range(100):
        loss, grads = training_tools.train_step(
            test_tokens,
            inputs_handler=token_feed,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer
        )
        print(f"Step {step} loss: {loss.numpy()}")
