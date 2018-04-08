from collections import Counter
import re


class EmbeddingLookup(object):

    def __init__(self, top_n=None):
        self._tokenizer = re.compile("\w+|\$[\d\.]+").findall
        self._top_n = top_n
        self._dictionary = None
        self._reverse = None

        self._unknown = "<unk>"

    def _get_top_n_tokens(self):
        top = self._dictionary.most_common(self._top_n)
        top = {item[0]: idx + 1 for idx, item in enumerate(top)}
        top[self._unknown] = self._top_n + 1
        return top

    def fit(self, corpus):
        tokens = [word for text in corpus for word in self._tokenizer(text.strip())]
        self._dictionary = Counter(tokens)

        if self._top_n is not None:
            self._dictionary = self._get_top_n_tokens()
        else:
            self._dictionary = dict(self._dictionary)

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
