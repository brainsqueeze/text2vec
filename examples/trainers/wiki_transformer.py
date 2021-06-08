from typing import Generator, List, Tuple, Union

import datasets
import tokenizers
from tokenizers import models
from tokenizers import decoders
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from text2vec.autoencoders import TransformerAutoEncoder
from text2vec.optimizer_tools import RampUpDecaySchedule


def train_tokenizer() -> Tuple[tokenizers.Tokenizer, Generator, int]:
    tokenizer = tokenizers.Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.decoder = decoders.WordPiece()

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),  # NFD unicode normalizer
        normalizers.Lowercase(),
        normalizers.StripAccents()
    ])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(individual_digits=False)
    ])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        pair="$A </s> [SEP] <s> $B:1",
        special_tokens=[("[SEP]", 1), ("<s>", 2), ("</s>", 3)]
    )

    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    tokenizer.train_from_iterator(
        batch_iterator(),
        trainer=trainers.WordPieceTrainer(
            vocab_size=10000,
            special_tokens=["<unk>", "[SEP]", "<s>", "</s>"]
        )
    )

    def generator():
        for record in dataset:
            if record['text'].strip() != '':
                yield record['text']

    return tokenizer, generator, len(dataset)


def main(save_path: str):
    tokenizer, data_gen, cardinality = train_tokenizer()
    tokenizer.save(path=f"{save_path}/tokenizer.json")

    with open(f"{save_path}/metadata.tsv", "w") as tsv:
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda s: s[-1]):
            # since tensorflow converts strings to bytes we will decode from UTF-8 here for display purposes
            tsv.write(f"{token}\n")

    def encode(x):
        def token_mapper(text: Union[str, List[str]]):
            text = text.numpy()

            if isinstance(text, np.ndarray):
                enc, dec = [], []
                for batch in tokenizer.encode_batch([(t.decode('utf8'), t.decode('utf8')) for t in text]):
                    enc_, dec_ = ' '.join(batch.tokens).split(' [SEP] ')
                    enc.append(enc_)
                    dec.append(dec_)
                return (enc, dec)

            text = text.decode('utf8')
            enc, dec = ' '.join(tokenizer.encode(text, pair=text).tokens).split(' [SEP] ')
            return (enc, dec)

        return tf.py_function(token_mapper, inp=[x], Tout=[tf.string, tf.string])

    data = tf.data.Dataset.from_generator(data_gen, output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string)))
    data = data.map(tf.strings.strip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data = data.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    model = TransformerAutoEncoder(
        max_sequence_len=512,
        embedding_size=256,
        token_hash=tokenizer.get_vocab(),
        input_keep_prob=0.7,
        hidden_keep_prob=0.5
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=RampUpDecaySchedule(embedding_size=256)))
    model.fit(x=data.prefetch(10).batch(16), epochs=1)
    checkpoint = tf.train.Checkpoint(Classifier=model, optimizer=model.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)

    # add word labels to the projector
    config = projector.ProjectorConfig()
    embeddings_config = config.embeddings.add()

    checkpoint_manager.save()
    reader = tf.train.load_checkpoint(save_path)
    embeddings_config.tensor_name = [key for key in reader.get_variable_to_shape_map() if "embedding" in key][0]
    embeddings_config.metadata_path = f"{save_path}/metadata.tsv"
    projector.visualize_embeddings(logdir=f"{save_path}", config=config)

    model.fit(
        x=data.prefetch(8).batch(16),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=save_path,
                write_graph=True,
                update_freq=cardinality // 100
            ),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: checkpoint_manager.save())
        ],
        epochs=1
    )

    tf.keras.models.save_model(model, filepath=f"{save_path}/saved_model", include_optimizer=False, save_format="tf")
    return model


if __name__ == '__main__':
    main(save_path='./')