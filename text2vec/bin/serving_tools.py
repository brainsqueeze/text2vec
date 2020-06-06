from text2vec.preprocessing.text import normalize_text, clean_and_split
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import re
import os


class Embedder(object):

    def __init__(self):
        log_dir = f"{os.environ['MODEL_PATH']}/frozen/1"
        self.__model = tf.saved_model.load(log_dir)

    def __get_sentences(self, text):
        data = [(sent, ' '.join(clean_and_split(normalize_text(sent)))) for sent in sent_tokenize(text)]
        data = [(orig, clean) for orig, clean in data if len(clean.split()) >= 5]
        original, clean = map(list, zip(*data))
        return original, clean

    def __normalize(self, vectors):
        assert isinstance(vectors, tf.Tensor)
        return tf.math.l2_normalize(vectors, axis=-1).numpy()

    def __doc_vector(self, doc):
        assert isinstance(doc, tf.Tensor)
        net_vector = tf.reduce_sum(doc, axis=0)
        return self.__normalize(net_vector)

    def __embed(self, corpus):
        assert isinstance(corpus, list)
        return self.__model.embed(corpus)

    def embed(self, text):
        assert isinstance(text, str)

        sentences, clean_sentences = self.__get_sentences(text)
        vectors = self.__embed(clean_sentences)
        return sentences, self.__normalize(vectors), self.__doc_vector(vectors)
