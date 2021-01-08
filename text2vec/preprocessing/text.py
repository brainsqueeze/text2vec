import re

TOKENIZER = re.compile(r"<\w+/>|\w+\'\w*|\w+|\$\d+|\d+|[,!?;.]")


def clean_and_split(text: str, compiled_pattern=TOKENIZER):
    """Splits the input text in tokens based on a boundary pattern.

    Parameters
    ----------
    text : str
        Input text
    compiled_pattern : regex pattern, optional
        Regular expression boundary pattern for tokenization, by default text2vec.preprocessing.text.TOKENIZER

    Returns
    -------
    list
        All RegEx tokens found by the boundary pattern.
    """

    text = text.lower().strip()
    if not hasattr(compiled_pattern, 'findall'):
        return text.split()
    return compiled_pattern.findall(text)


def replace_money_token(text):
    """Replace monetary mentions with a <money/> tag.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    return re.sub(r"(\$|€)\d*((\.|,)*\d*)*", "<money/>", text, re.M | re.I)


def replace_urls_token(text):
    """Replace URLs mentioned in text with a <url/> tag.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    text = re.sub(r"^https?://.*[\r\n]*", "<url/>", text, re.M | re.I)
    return re.sub(r"http\S+(\s)*(\w+\.\w+)*", "<url/>", text, re.M | re.I)


def fix_unicode_quotes(text):
    """Replace some unicode objects with their ASCII equivalent.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    text = re.sub(u'[\u201c\u201d]', '"', text, re.M | re.I)
    text = re.sub(u'[\u2018\u2019\u0027]', "'", text, re.M | re.I)
    return re.sub(u'[\u2014]', "-", text, re.M | re.I)


def format_large_numbers(text):
    """Standardize large number amounts.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    text = re.sub(r"(?<!\d)\$?\d{1,3}(?=(,\d{3}|\s))", r" \g<0> ", text)  # pad commas in large numerical values
    return re.sub(r"(\d+)?,(\d+)", r"\1\2", text)  # remove commas from large numerical values


def pad_punctuation(text):
    """Pad spaces around punctuation to treat as a separate token.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    text = re.sub(r"([!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~])", r" \1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return re.sub(r"(<)\s(\w+)\s(/)\s(>)", r"\1\2\3\4", text, re.I | re.M)  # keep special tokens intact


def normalize_text(text):
    """
    Regular expression clean-ups of the text.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
    """

    text = text.lower().strip().replace("\n", " ").replace("\r", "")

    text = replace_money_token(text)
    text = replace_urls_token(text)
    text = fix_unicode_quotes(text)
    text = format_large_numbers(text)
    text = pad_punctuation(text)
    return text.strip()
