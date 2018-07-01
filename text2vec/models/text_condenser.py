from text2vec.models.embedding import TextAttention
import tensorflow as tf
import numpy as np

import nltk.data

from itertools import groupby

import pickle
import json
import os


class Embedder(object):

    def __init__(self, use_gpu=False):
        self._root = os.path.dirname(os.path.abspath(__file__)) + "/../../" + os.environ["MODEL_PATH"]

        num_hidden, self._max_seq_len, vocab_size, attention_size, logdir = self._get_metadata()

        with open(logdir + "/lookup.pkl", "rb") as pf:
            self._lookup = pickle.load(pf)

        self._seq_input = tf.placeholder(dtype=tf.int32, shape=[None, self._max_seq_len])
        self._keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

        self._model = TextAttention(
            input_x=self._seq_input,
            embedding_size=100,
            vocab_size=vocab_size,
            keep_prob=self._keep_prob,
            num_hidden=num_hidden,
            attention_size=attention_size,
            is_training=False
        )

        if use_gpu:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            sess_config = tf.ConfigProto(
                gpu_options=gpu_options,
                allow_soft_placement=True,
                log_device_placement=False
            )
            self._session = tf.Session(config=sess_config)
        else:
            self._session = tf.Session()

        self._session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(logdir)
        saver.restore(self._session, checkpoint.model_checkpoint_path)

    def _get_metadata(self):
        log_directory = self._root

        with open(log_directory + "/model.json", "r") as file_sys:
            meta_data = json.load(file_sys)

        num_hidden = meta_data["embeddingDimensions"]
        max_seq_length = meta_data["maxSequenceLength"]
        vocab_size = meta_data["vocabSize"]
        attention_size = meta_data["attentionWeightDim"]

        return num_hidden, max_seq_length, vocab_size, attention_size, log_directory

    def _pad_sequence(self, sequence):
        sequence = np.array(sequence, dtype=np.int32)[:self._max_seq_len]
        difference = self._max_seq_len - sequence.shape[0]
        pad = np.zeros((difference,), dtype=np.int32)
        return np.concatenate((sequence, pad))

    def process_input(self, text_input):
        x = self._lookup.transform(corpus=text_input)
        return np.array([self._pad_sequence(seq) for seq in x], dtype=np.int32)

    def embed(self, x):
        return self._session.run(self._model.context, feed_dict={self._seq_input: x})


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
        return self._sentence_tokenizer.tokenize(input_text)
