import tensorflow as tf
import numpy as np

import nltk.data

from itertools import groupby

import pickle
import os


class Embedder(object):

    def __init__(self):
        self.__log_dir = os.path.dirname(os.path.abspath(__file__)) + "/../../" + os.environ["MODEL_PATH"]

        with open(self.__log_dir + "/lookup.pkl", "rb") as pf:
            self.lookup = pickle.load(pf)
            self.max_seq_len = self.lookup.max_sequence_length

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess_config = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True,
            log_device_placement=False
        )
        graph = self.__load_model()
        self.__session = tf.Session(graph=graph, config=sess_config)

    def __load_model(self):
        name = "embedder"
        model_dir = self.__log_dir + "/saved"
        with tf.gfile.GFile(model_dir + '/frozen_model.pb', 'rb') as f:
            graph_def_optimized = tf.GraphDef()
            graph_def_optimized.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def_optimized, name=name)

        embed_tensor_names = [
            op.name for op in graph.get_operations()
            if 'context' in op.name and 'attention' in op.name and op.type != 'Const']

        assert len(embed_tensor_names) == 1
        self.seq_input = graph.get_tensor_by_name(f'{name}/sequence-input:0')
        self.embedding = graph.get_tensor_by_name(embed_tensor_names[0] + ':0')
        return graph

    def _pad_sequence(self, sequence):
        sequence = np.array(sequence, dtype=np.int32)[:self.max_seq_len]
        difference = self.max_seq_len - sequence.shape[0]
        pad = np.zeros((difference,), dtype=np.int32)
        return np.concatenate((sequence, pad))

    def process_input(self, text_input):
        x = self.lookup.transform(corpus=text_input)
        return np.array([self._pad_sequence(seq) for seq in x], dtype=np.int32)

    def embed(self, x):
        return self.__session.run(self.embedding, feed_dict={self.seq_input: x})


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
