from text2vec.preprocessing import clean_and_split
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import re
import os


class Embedder(object):

    def __init__(self):
        log_dir = f"{os.environ['MODEL_PATH']}/frozen/1"
        self.__model = tf.saved_model.load(log_dir)

    @staticmethod
    def normalize_text(text):
        text = text.lower().strip().replace("\n", " ").replace("\r", "")

        text = re.sub(r"(\$|€)\d*((\.|,)*\d*)*", "<money/>", text, re.M | re.I)  # replace money amounts with <money/>
        text = re.sub(r"^https?://.*[\r\n]*", "<url/>", text, re.M | re.I)  # replace URLs
        text = re.sub(r"http\S+(\s)*(\w+\.\w+)*", "<url/>", text, re.M | re.I)  # replace URLs

        # fix unicode quotes and dashes
        text = re.sub(u'[\u201c\u201d]', '"', text, re.M | re.I)
        text = re.sub(u'[\u2018\u2019\u0027]', "'", text, re.M | re.I)
        text = re.sub(u'[\u2014]', "-", text, re.M | re.I)

        text = re.sub(r"(?<!\d)\$?\d{1,3}(?=(,\d{3}|\s))", r" \g<0> ", text)  # pad commas in large numerical values
        text = re.sub(r"(\d+)?,(\d+)", r"\1\2", text)  # remove commas from large numerical values

        text = re.sub(r"([!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~])", r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"(<)\s(\w+)\s(/)\s(>)", r"\1\2\3\4", text, re.I | re.M)  # keep special tokens intact
        return text.strip()

    def __get_sentences(self, text):
        data = [(sent, ' '.join(clean_and_split(self.normalize_text(sent)))) for sent in sent_tokenize(text)]
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
