from typing import Tuple
import os

import datasets
import tokenizers
from tokenizers import models
from tokenizers import decoders
from tokenizers import normalizers
from tokenizers import pre_tokenizers
from tokenizers import trainers

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras import backend as K
from tensorboard.plugins import projector

from text2vec.autoencoders import LstmAutoEncoder
from text2vec.optimizer_tools import RampUpDecaySchedule

os.environ["TOKENIZERS_PARALLELISM"] = "true"
root = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_SIZE = 128
MAX_SEQUENCE_LENGTH = 512


def train_tokenizer() -> Tuple[tokenizers.Tokenizer, tf.data.Dataset]:
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

    dataset = datasets.load_dataset("multi_news", split="train")

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            for key in dataset.features:
                yield dataset[i: i + batch_size][key]

    tokenizer.train_from_iterator(
        batch_iterator(),
        trainer=trainers.WordPieceTrainer(
            vocab_size=10000,
            special_tokens=["<unk>", "[SEP]", "<s>", "</s>"]
        )
    )

    tokenizer.enable_truncation(2 * MAX_SEQUENCE_LENGTH + 3)  # 2 for the [SEP], <s>, </s> tokens
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="$A",
        pair="$A:0 [SEP] <s> $B:1 </s>",
        special_tokens=[
            ("[SEP]", 1),
            ("<s>", 2),
            ("</s>", 3)
        ]
    )

    def generator():
        for record in dataset:
            if record["document"] and record["summary"]:
                enc, dec = ' '.join(tokenizer.encode(
                    record["document"],
                    pair=record["summary"]
                ).tokens).split(' [SEP] ', maxsplit=2)

                if enc.strip() != "" and dec != "":
                    yield enc, dec

    data = tf.data.Dataset.from_generator(
        generator,
        output_signature=(tf.TensorSpec(shape=(None), dtype=tf.string), tf.TensorSpec(shape=(None), dtype=tf.string))
    )
    return tokenizer, data


def main(save_path: str):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    tokenizer, data = train_tokenizer()
    with open(f"{save_path}/metadata.tsv", "w", encoding="utf8") as tsv:
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda s: s[-1]):
            tsv.write(f"{token.encode('utf8')}\n")

    model = LstmAutoEncoder(
        max_sequence_len=MAX_SEQUENCE_LENGTH,
        embedding_size=EMBEDDING_SIZE,
        token_hash=tokenizer.get_vocab(),
        input_drop_rate=0.3,
        hidden_drop_rate=0.5
    )

    scheduler = RampUpDecaySchedule(EMBEDDING_SIZE, warmup_steps=4000)
    model.compile(optimizer=optimizers.Adam(scheduler(0).numpy()))
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

    model.fit(
        x=data.prefetch(8).shuffle(10_000).batch(64),
        callbacks=[
            callbacks.TensorBoard(log_dir=save_path, write_graph=True, update_freq=100),
            callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: checkpoint_manager.save(),
                on_batch_end=lambda batch, logs: K.set_value(
                    model.optimizer.lr,
                    K.get_value(scheduler(model.optimizer.iterations))
                )
            )
        ],
        epochs=1
    )

    model.save(
        filepath=f"{save_path}/saved_model",
        save_format="tf",
        include_optimizer=False,
        signatures={"serving_default": model.embed, "token_embed": model.token_embed}
    )
    tokenizer.save(path=f"{save_path}/tokenizer.json")
    return model


if __name__ == '__main__':
    main(save_path=f'{root}/../../multi_news_t2v_sequential')
