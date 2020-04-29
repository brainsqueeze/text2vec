from collections import Counter
import re

import tensorflow as tf

TOKENIZER = re.compile(r"<\w+/>|\w+\'\w*|\w+|\$\d+|\d+|[,!?;.]")


def clean_and_split(text, compiled_pattern=TOKENIZER):
    """
    Splits input text into tokens based on a boundary pattern.
    :param corpus: text (str)
    :param compiled_pattern: compiled regular expression which defines token boundaries (regex Pattern, optional)
    :return: list of tokens split per the regex expression (list)
    """

    assert isinstance(compiled_pattern, re.Pattern)
    text = text.lower().strip()
    return compiled_pattern.findall(text)


def get_top_tokens(corpus, n_top=1000):
    """
    Builds the token mapping which is used to initialize the word embeddings in the model.
    Get the most frequent terms which appear in the training corpus.
    :param corpus: list of strings (list)
    :param n_top: the number of most frequent tokens (int, optional, default=1000)
    :return: token->integer lookup, maximum sequence length (dict, int)
    """

    corpus = [clean_and_split(text) for text in corpus]
    max_sequence_length = max(map(len, corpus))
    lookup = Counter([token for sequence in corpus for token in sequence])
    lookup = lookup.most_common(n_top)

    hash_map = {key: idx + 2 for idx, (key, value) in enumerate(lookup)}
    hash_map["<s>"] = 0
    hash_map["</s>"] = 1
    return hash_map, max_sequence_length


def normalize_text(text):
    """
    Regular expression clean-ups of the text.
    :param corpus: text (str)
    :return: cleaned text (str)
    """

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
