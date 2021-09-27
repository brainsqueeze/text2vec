from typing import Generator, List, Tuple, Union
import os

import datasets
import tokenizers
from tokenizers import models
from tokenizers import decoders
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import processors
from tokenizers import trainers
from nltk.tokenize import PunktSentenceTokenizer

import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from text2vec.autoencoders import TransformerAutoEncoder
from text2vec.optimizer_tools import RampUpDecaySchedule
from text2vec.training_tools import ServingModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"
sent_tokenizer = PunktSentenceTokenizer().tokenize


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

    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

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
                for sentence in sent_tokenizer(record['text']):
                    yield sentence

    data = tf.data.Dataset.from_generator(generator, output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string)))
    data = data.map(tf.strings.strip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return tokenizer, data


def main(save_path: str):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    tokenizer, data = train_tokenizer()
    tokenizer.enable_truncation(2 * 512 + 1)  # encoding + decoding + [SEP] token

    with open(f"{save_path}/metadata.tsv", "w") as tsv:
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda s: s[-1]):
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

    model = TransformerAutoEncoder(
        max_sequence_len=512,
        embedding_size=128,
        token_hash=tokenizer.get_vocab(),
        input_keep_prob=0.7,
        hidden_keep_prob=0.5
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=RampUpDecaySchedule(embedding_size=128)))
    checkpoint = tf.train.Checkpoint(Classifier=model, optimizer=model.optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)

    # add word labels to the projector
    config = projector.ProjectorConfig()
    # pylint: disable=no-member
    embeddings_config = config.embeddings.add()

    checkpoint_manager.save()
    reader = tf.train.load_checkpoint(save_path)
    embeddings_config.tensor_name = [key for key in reader.get_variable_to_shape_map() if "embedding" in key][0]
    embeddings_config.metadata_path = f"{save_path}/metadata.tsv"
    projector.visualize_embeddings(logdir=save_path, config=config)

    data = data.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    model.fit(
        x=data.prefetch(8).batch(64),
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=save_path,
                write_graph=True,
                update_freq=100
            ),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: checkpoint_manager.save())
        ],
        epochs=1
    )

    # tf.keras.models.save_model(model, filepath=f"{save_path}/saved_model", include_optimizer=False, save_format="tf")
    serve_model = ServingModel(
        tokenizer=model.tokenizer,
        embed_layer=model.embed_layer,
        encode_layer=model.encode_layer
    )
    tf.saved_model.save(
        obj=serve_model,
        export_dir=f"{save_path}/saved_model",
        signatures={"serving_default": serve_model.embed, "token_embed": serve_model.token_embed}
    )
    tokenizer.save(path=f"{save_path}/tokenizer.json")
    return model


if __name__ == '__main__':
    main(save_path='./wiki_t2v')
