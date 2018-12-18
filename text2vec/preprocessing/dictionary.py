from collections import Counter
import re

import math


class EmbeddingsNotFitError(Exception):
    pass


class EmbeddingLookup(object):

    def __init__(self, top_n=None, use_tf_idf_importance=False):
        self._tokenizer = re.compile(r"\w+|\$[\d\.]+").findall
        self._top_n = top_n
        self._use_tf_idf = use_tf_idf_importance

        self._dictionary = None
        self._reverse = None
        self._unknown = "<unk>"
        self._end_sequence = "<eos>"
        self._begin_sequence = "<bos>"

        self.__error_message = "You must fit the lookup first, either by calling the fit or fit_transform methods."

    def _build_lookup(self, terms):
        top = {key: idx + 2 for idx, (key, value) in enumerate(terms)}
        # top[self._unknown] = len(top) + 1
        top[self._unknown] = max(top.values()) + 1  # add the <unk> token to the embedding lookup
        top[self._end_sequence] = 0
        top[self._begin_sequence] = 1
        return top

    def _get_top_n_tokens_tf_idf(self, corpus):
        # create set of tokens for each document
        doc_lookups = [set(self._tokenizer(document)) for document in corpus]

        inverse_df = []
        for i in range(len(corpus)):
            document = doc_lookups[i]

            # for each document, find all of the unique words
            # these will be counted up at the end (unique occurrences) to get document frequencies
            present_tokens = document.intersection(self._dictionary)
            inverse_df.extend(present_tokens)

        # compute the smooth TF-IDF values
        inverse_df = {k: math.log(1 + (1 + len(corpus)) / (1 + v)) for k, v in Counter(inverse_df).items()}
        total_dfs = sum(self._dictionary.values())
        tf_idf = {token: (self._dictionary[token] / total_dfs) * inverse_df[token] for token in inverse_df}

        # sort by TF-IDF values, descending, then build the dictionary using these tokens
        sort_terms = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        return self._build_lookup(terms=sort_terms)

    def _get_top_n_tokens(self):
        top = self._dictionary.most_common(self._top_n)
        return self._build_lookup(terms=top)

    def fit(self, corpus, vocab_set=None):
        if vocab_set:
            tokens = [word for text in corpus for word in self._tokenizer(text.lower().strip()) if word in vocab_set]
        else:
            tokens = [word for text in corpus for word in self._tokenizer(text.lower().strip())]
        self._dictionary = Counter(tokens)

        if self._top_n is not None and self._top_n < len(self._dictionary):
            if self._use_tf_idf:
                self._dictionary = self._get_top_n_tokens_tf_idf(corpus)
            else:
                self._dictionary = self._get_top_n_tokens()
        else:
            sort_terms = sorted(self._dictionary.items(), key=lambda x: x[1], reverse=True)
            self._dictionary = self._build_lookup(terms=sort_terms)

        self._reverse = {v: k for k, v in sorted(self._dictionary.items(), key=lambda x: x[1])}
        return self

    def transform(self, corpus):
        if self._dictionary is None:
            raise EmbeddingsNotFitError(self.__error_message)

        corpus = [
            [
                self._dictionary[word] if word in self._dictionary else self._dictionary[self._unknown]
                for word in self._tokenizer(text.lower().strip())
            ] for text in corpus]
        return corpus

    def fit_transform(self, corpus, vocab_set=None):
        self.fit(corpus, vocab_set=vocab_set)
        return self.transform(corpus)

    def __getitem__(self, key):
        if not isinstance(self._dictionary, dict):
            raise EmbeddingsNotFitError(self.__error_message)
        return self._dictionary.get(key, None)

    def __contains__(self, item):
        return item in self._dictionary

    def __len__(self):
        return len(self._dictionary)

    @property
    def reverse(self):
        return self._reverse
