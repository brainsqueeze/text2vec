import os
from typing import List, Union

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
from text2vec.autoencoders import TransformerAutoEncoder

root = os.path.dirname(os.path.abspath(__file__))


def train_tokenizer() -> tokenizers.Tokenizer:
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
        special_tokens=[
            ("[SEP]", 1),
            ("<s>", 2),
            ("</s>", 3)
        ]
    )

    # dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train+test+validation")
    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

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

    return tokenizer, generator


def main():
    tokenizer, data_gen = train_tokenizer()

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
        embedding_size=128,
        token_hash=tokenizer.get_vocab(),
        input_keep_prob=0.7,
        hidden_keep_prob=0.5
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
        run_eagerly=True
    )
    model.fit(x=data.prefetch(10).batch(16), epochs=1)

    model(['here is a sentence', 'try another one'])
    model.predict(['here is a sentence', 'try another one'])
    return model


if __name__ == '__main__':
    main()
