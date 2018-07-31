from collections import Counter
import re

import math


class EmbeddingLookup(object):

    def __init__(self, top_n=None, use_tf_idf_importance=False):
        self._tokenizer = re.compile("\w+|\$[\d\.]+").findall
        self._top_n = top_n
        self._use_tf_idf = use_tf_idf_importance

        self._dictionary = None
        self._reverse = None
        self._unknown = "<unk>"

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
        inverse_df = {k: math.log(1 + len(corpus)) / (1 + v) for k, v in Counter(inverse_df).items()}
        total_dfs = sum(self._dictionary.values())
        tf_idf = {token: (self._dictionary[token] / total_dfs) * inverse_df[token] for token in self._dictionary}

        # sort by TF-IDF values, descending, then build the dictionary using these tokens
        sort_terms = sorted(tf_idf, key=lambda x: x[1], reverse=True)[:self._top_n]
        top = {item[0]: idx + 1 for idx, item in enumerate(sort_terms)}
        top[self._unknown] = len(top) + 1
        return top

    def _get_top_n_tokens(self):
        top = self._dictionary.most_common(self._top_n)
        top = {item[0]: idx + 1 for idx, item in enumerate(top)}
        top[self._unknown] = len(top) + 1
        return top

    def fit(self, corpus):
        tokens = [word for text in corpus for word in self._tokenizer(text.strip())]
        self._dictionary = Counter(tokens)

        if self._top_n is not None:
            if self._use_tf_idf and self._top_n > len(self._dictionary):
                self._dictionary = self._get_top_n_tokens_tf_idf(corpus)
            else:
                self._dictionary = self._get_top_n_tokens()
        else:
            self._dictionary = {item[0]: idx + 1 for idx, item in enumerate(self._dictionary.items())}

            # add the <unk> token to the embedding lookup
            size = len(self._dictionary)
            self._dictionary[self._unknown] = size + 1

        self._reverse = {v: k for k, v in self._dictionary.items()}
        return self

    def transform(self, corpus):
        if self._dictionary is None:
            raise AttributeError("You must fit the lookup first, either by calling the fit or fit_transform methods.")

        corpus = [
            [
                self._dictionary[word] if word in self._dictionary else self._dictionary[self._unknown]
                for word in self._tokenizer(text.strip())
            ] for text in corpus]
        return corpus

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def __getitem__(self, key):
        if not isinstance(self._dictionary, dict):
            raise TypeError("You must fit the lookup first, either by calling the fit or fit_transform methods.")
        return self._dictionary.get(key, None)

    @property
    def reverse(self):
        return self._reverse
