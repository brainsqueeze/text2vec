from collections import Counter
import re

TOKENIZER = re.compile(r"\w+\'\w*|\w+|\$\d+|\d+|[,!?;.]")


def clean_and_split(text):
    text = text.lower().strip()
    return TOKENIZER.findall(text)


def get_top_tokens(corpus, n_top=1000):
    corpus = [clean_and_split(text) for text in corpus]
    max_sequence_length = max(map(len, corpus))
    lookup = Counter([token for sequence in corpus for token in sequence])
    lookup = lookup.most_common(n_top)

    hash_map = {key: idx + 2 for idx, (key, value) in enumerate(lookup)}
    hash_map["<s>"] = 0
    hash_map["</s>"] = 1
    return hash_map, max_sequence_length
