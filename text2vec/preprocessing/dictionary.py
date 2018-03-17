from collections import Counter
from nltk.tokenize import word_tokenize
import os


class EmbeddingLookup(object):

    def __init__(self, path):
        self.path = path
        self.files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        self._split_combine()
        self.unknown = "<unk>"

    def _split_combine(self):
        tokens = []

        for file in self.files:
            with open(self.path + file, "r", encoding="latin1") as f:
                lines = f.readlines()

            for line in lines:
                tokens.extend(word_tokenize(line))

        self.corpus = Counter(tokens)
        return self

    def get_top_n_tokens(self, n=1000):
        top = self.corpus.most_common(n)
        top_n = {item[0]: idx + 1 for idx, item in enumerate(top)}
        return {**top_n, **{self.unknown: max(top_n.values()) + 1}}

    def token_replace_id(self, top_n=1000):
        top = self.get_top_n_tokens(n=top_n)
        corpus = []

        for file in self.files:
            with open(self.path + file, "r", encoding="latin1") as f:
                corpus.extend([
                    [top[token] if token in top else top[self.unknown] for token in word_tokenize(line)]
                    for line in f.readlines()
                ])
        return corpus
