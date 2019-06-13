import tensorflow as tf
import numpy as np

from text2vec.preprocessing import clean_and_split
import nltk.data

from itertools import groupby

import os


class Embedder(object):

    def __init__(self):
        self.__log_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../" + os.environ["MODEL_PATH"]

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess_config = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False
        )
        self.__session = tf.Session(graph=tf.Graph(), config=sess_config)
        self.__load_model()

        self.predict("")

    def __load_model(self):
        model = tf.compat.v1.saved_model.loader.load(
            self.__session,
            tags=[tf.saved_model.tag_constants.SERVING],
            export_dir=self.__log_dir + "/saved"
        )

        def_key = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        graph = self.__session.graph
        self.__seq_input = graph.get_tensor_by_name(model.signature_def[def_key].inputs["sequences"].name)
        self.__output = graph.get_tensor_by_name(model.signature_def[def_key].outputs["embedding"].name)
        return graph

    def __embed(self, corpus):
        assert isinstance(corpus, list)
        epsilon = 1e-8

        vectors = self.__session.run(self.__output, feed_dict={self.__seq_input: corpus})
        vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + epsilon)
        return vectors

    def predict(self, text):
        corpus = [clean_and_split(text)]
        vectors = self.__embed(corpus)
        return vectors[0]  # only 1 text at a time so the batch size is 1

    def embed(self, corpus):
        assert isinstance(corpus, list)

        corpus = [clean_and_split(text) for text in corpus]
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
