from collections import Counter
import re

import math


class EmbeddingsNotFitError(Exception):
    pass


class EmbeddingLookup(object):

    def __init__(self, top_n=None, preprocessor=None, use_tf_idf_importance=False):
        self._tokenizer = re.compile(r"\w+\'\w*|\w+|\$\d+|\d+|[,!?;.]").findall
        self._top_n = top_n
        self._use_tf_idf = use_tf_idf_importance

        if preprocessor is None:
            self.__process = self.default_processor
        else:
            self.__process = preprocessor

        self.__dictionary = None
        self.__reverse = None
        self.unknown = "<unk>"
        self.end_sequence = "<eos>"
        self.begin_sequence = "<bos>"
        self.__max_sequence_length = 0

        self.__error_message = "You must fit the lookup first, either by calling the fit or fit_transform methods."

    @staticmethod
    def default_processor(text):
        return text.lower().strip()

    def _build_lookup(self, terms):
        top = {key: idx + 2 for idx, (key, value) in enumerate(terms)}
        top[self.unknown] = max(top.values()) + 1  # add the <unk> token to the embedding lookup
        top[self.end_sequence] = 0
        top[self.begin_sequence] = 1
        return top

    def _get_top_n_tokens_tf_idf(self, corpus):
        # create set of tokens for each document
        doc_lookups = [set(self._tokenizer(self.__process(document))) for document in corpus]

        inverse_df = []
        for i in range(len(corpus)):
            document = doc_lookups[i]

            # for each document, find all of the unique words
            # these will be counted up at the end (unique occurrences) to get document frequencies
            present_tokens = document.intersection(self.__dictionary)
            inverse_df.extend(present_tokens)

        # compute the smooth TF-IDF values
        inverse_df = {k: math.log(1 + (1 + len(corpus)) / (1 + v)) for k, v in Counter(inverse_df).items()}
        total_dfs = sum(self.__dictionary.values())
        tf_idf = {token: (self.__dictionary[token] / total_dfs) * inverse_df[token] for token in inverse_df}

        # sort by TF-IDF values, descending, then build the dictionary using these tokens
        sort_terms = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        return self._build_lookup(terms=sort_terms)

    def _get_top_n_tokens(self):
        top = self.__dictionary.most_common(self._top_n)
        return self._build_lookup(terms=top)

    def _make_tokens(self, corpus, vocab_set=None):
        corpus = [self._tokenizer(self.__process(text)) for text in corpus]
        max_corpus_sequence_length = max(map(len, corpus))
        if max_corpus_sequence_length > self.__max_sequence_length:
            self.__max_sequence_length = max_corpus_sequence_length

        if vocab_set:
            return [word for text in corpus for word in text if word in vocab_set]
        return [word for text in corpus for word in text]

    def fit(self, corpus, vocab_set=None):
        tokens = self._make_tokens(corpus=corpus, vocab_set=vocab_set)
        self.__dictionary = Counter(tokens)

        if self._top_n is not None and self._top_n < len(self.__dictionary):
            if self._use_tf_idf:
                self.__dictionary = self._get_top_n_tokens_tf_idf(corpus)
            else:
                self.__dictionary = self._get_top_n_tokens()
        else:
            sort_terms = sorted(self.__dictionary.items(), key=lambda x: x[1], reverse=True)
            self.__dictionary = self._build_lookup(terms=sort_terms)

        self.__reverse = {v: k for k, v in sorted(self.__dictionary.items(), key=lambda x: x[1])}
        return self

    def batch_fit(self, corpus_generator, vocab_set=None):
        for corpus_batch in corpus_generator:
            tokens = self._make_tokens(corpus=corpus_batch, vocab_set=vocab_set)
            if self.__dictionary is None:
                self.__dictionary = Counter(tokens)
            elif isinstance(self.__dictionary, Counter):
                self.__dictionary.update(tokens)
            else:
                raise TypeError
        if self._top_n is not None and self._top_n < len(self.__dictionary):
            self.__dictionary = self._get_top_n_tokens()
        else:
            sort_terms = sorted(self.__dictionary.items(), key=lambda x: x[1], reverse=True)
            self.__dictionary = self._build_lookup(terms=sort_terms)

        self.__reverse = {v: k for k, v in sorted(self.__dictionary.items(), key=lambda x: x[1])}
        return

    def transform(self, corpus):
        if self.__dictionary is None:
            raise EmbeddingsNotFitError(self.__error_message)

        corpus = [
            [
                self.__dictionary[word] if word in self.__dictionary else self.__dictionary[self.unknown]
                for word in self._tokenizer(text)
            ] for text in map(self.__process, corpus)
            if self._tokenizer(text)
        ]
        return corpus

    def fit_transform(self, corpus, vocab_set=None):
        self.fit(corpus, vocab_set=vocab_set)
        return self.transform(corpus)

    def __getitem__(self, key):
        if not isinstance(self.__dictionary, dict):
            raise EmbeddingsNotFitError(self.__error_message)
        return self.__dictionary.get(key, None)

    def __contains__(self, item):
        return item in self.__dictionary

    def __len__(self):
        return len(self.__dictionary)

    @property
    def reverse(self):
        return self.__reverse

    @property
    def max_sequence_length(self):
        return self.__max_sequence_length
