import tensorflow as tf
import numpy as np

from text2vec.preprocessing import clean_and_split
import nltk.data

from itertools import groupby

import os


class Embedder(object):

    def __init__(self):
        log_dir = f"{os.environ['MODEL_PATH']}/frozen/1"
        model = tf.saved_model.load(log_dir)

        self.__embedding_model = model.signatures["serving_default"]
        self.__word_embedding_model = model.signatures["token_embed"]

    def __embed(self, corpus):
        assert isinstance(corpus, list)

        vectors = self.__embedding_model(tf.constant(corpus))["output_0"]
        vectors = tf.linalg.l2_normalize(vectors, axis=-1)
        return vectors.numpy()

    def predict(self, text):
        corpus = [' '.join(clean_and_split(text))]
        vectors = self.__embed(corpus)
        return vectors[0]  # only 1 text at a time so the batch size is 1

    def embed(self, corpus):
        assert isinstance(corpus, list)

        corpus = [' '.join(clean_and_split(text)) for text in corpus]
        vectors = self.__embed(corpus)
        return vectors


class TextHandler(object):

    def __init__(self):
        self._sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    @staticmethod
    def _paragraph_tokenizer(input_text):
        result = [
            ''.join(line_iteration)
            for group_separator, line_iteration in groupby(input_text.splitlines(True), key=str.isspace)
            if not group_separator
        ]
        return result

    def split_paragraphs(self, input_text):
        return self._paragraph_tokenizer(input_text)

    def split_sentences(self, input_text):
        return self._sentence_tokenizer.tokenize(input_text.replace(";", "."))
