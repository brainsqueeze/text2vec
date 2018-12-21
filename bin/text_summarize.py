from text2vec.models.text_condenser import TextHandler, Embedder
import numpy as np

import time
import json

from flask import Flask, request, Response
from flask_cors import cross_origin
import tornado
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
import tornado.autoreload
from tornado.options import parse_command_line

app = Flask(__name__)
th = TextHandler()
e = Embedder()


def responder(results, error, message):
    assert isinstance(results, dict)
    results["message"] = message
    results = json.dumps(results, indent=2)

    return Response(
        response=results,
        status=error,
        mimetype="application/json"
    )


def cosine_similarity_sort(net_vector, embedding_matrix, attend=False):
    """
    Computes the cosine similarity scores and then returns
    the sorted results
    :param net_vector: the context vector for the entire document (ndarray)
    :param embedding_matrix: the context vectors (row vectors) for each constituent body of text (ndarray)
    :param attend: set flag to get the self-attention for the embeddings (bool, optional)
    :return: (sorted order of documents, cosine similarity scores)
    """

    assert net_vector.shape[0] == 1
    assert net_vector.shape[-1] == embedding_matrix.shape[-1]

    if attend:
        embedding_matrix = attention(embedding_matrix, embedding_matrix, embedding_matrix)

    net_vector /= np.linalg.norm(net_vector, axis=1, keepdims=True)
    embedding_matrix /= np.linalg.norm(embedding_matrix, axis=1, keepdims=True)

    net_vector[np.isnan(net_vector)] = 0
    embedding_matrix[np.isnan(embedding_matrix)] = 0

    similarity = np.dot(net_vector, embedding_matrix.T)
    similarity = np.clip(similarity, -1, 1)
    sort = np.argsort(1 - similarity, axis=1)[0]

    return sort, similarity.flatten()[sort]


def angle_from_cosine(cosine_similarity):
    """
    Computes the angles in degrees from cosine similarity scores
    :param cosine_similarity: (ndarray)
    :return: angles in degrees (ndarray)
    """

    return np.arccos(cosine_similarity) * (180 / np.pi)


def choose(sentences, scores, embeddings):
    """
    Selects the best constituent texts from the similarity scores
    :param sentences: array of the input texts, sorted by scores (ndarray)
    :param scores: cosine similarity scores, sorted (ndarray)
    :param embeddings: embedding matrix for input texts, sorted by scores (ndarray)
    :return: best sentences sorted, best scores sorted, best embeddings sorted
    """

    if scores.shape[0] == 1:
        return sentences, scores, embeddings

    angles = angle_from_cosine(scores)
    likelihood = np.exp(-angles) / np.exp(-angles).sum()
    threshold = likelihood.mean()
    cut = likelihood >= threshold
    return sentences[cut], scores[cut], embeddings[cut]


def text_pass_filter(texts, texts_embeddings, net_vector):
    """
    Runs the scoring + filtering process on input texts
    :param texts: input texts (ndarray)
    :param texts_embeddings: context embedding matrix for input texts (ndarray)
    :param net_vector: the context vector for the entire document (ndarray)
    :return: best sentences sorted, best scores sorted, best embeddings sorted
    """

    sorted_order, scores = cosine_similarity_sort(
        net_vector=net_vector,
        embedding_matrix=texts_embeddings,
        attend=False
    )

    texts = np.array(texts)[sorted_order]
    filtered_texts, filtered_scores, filtered_embeddings = choose(
        sentences=texts,
        scores=scores,
        embeddings=texts_embeddings[sorted_order]
    )

    return filtered_texts, filtered_scores, filtered_embeddings


def softmax(logits):
    """
    Computes the softmax function along rows
    of the incoming logits matrix
    :param logits: (ndarray)
    :return: array of the same shape as logits (ndarray)
    """

    soft = np.exp(logits)
    soft[np.isinf(soft)] = 1e10
    soft /= np.sum(soft, axis=0)
    soft = np.clip(soft, 0.0, 1.0)
    return soft


def attention(queries, keys, values, mask_future=True):
    """
    Computes the dot-product attention, future masking is turned
    on by default
    :param queries: dimensions (C, D) (ndarray)
    :param keys: dimensions (N, D) (ndarray)
    :param values: dimensions (N, M) (ndarray)
    :param mask_future: (bool)
    :return: (ndarray)
    """

    numerator = np.dot(queries, keys.T)
    denominator = np.sqrt(keys.shape[-1])

    if mask_future:
        upper = (1 + 1e9) * np.triu(np.ones_like(numerator), k=1)  # strictly upper triangle, no diagonal
        mask = 1 - upper
        numerator *= mask

    scores = numerator / denominator
    scores = softmax(scores)
    return np.dot(scores, values)


def normal(vectors):
    """
    Computes the normal to a vector bundle
    :param vectors: dimensions (N, C) (ndarray)
    :return: (C) (ndarray)
    """

    n = np.sum(vectors, axis=0)
    n /= np.linalg.norm(n, axis=0)
    return n


def attended_norm(vectors):
    """
    Runs self-attention before computing the normal vector
    :param vectors: dimensions (N, C) (ndarray)
    :return: (C) (ndarray)
    """

    vectors = attention(queries=vectors, keys=vectors, values=vectors)
    return normal(vectors)


@app.route('/condense', methods=['POST', 'GET'])
@cross_origin(origins=['*'], allow_headers=['Content-Type', 'Authorization'])
def compute():
    """
    Main Flask handler function
    :return: (Flask response object)
    """

    j = request.get_json()
    if j is None:
        j = request.args
    if not j:
        j = request.form

    st = time.time()
    body = j.get("body", "")
    if not body:
        results = {
            "elapsed_time": time.time() - st,
            "data": None
        }
        return responder(results=results, error=400, message="No text provided")

    # get the embedding vectors for each sentence in the document
    sentences = [
        sentence for sent in th.split_paragraphs(body)
        for sentence in th.split_sentences(sent)
        if len(sentence.split()) > 3
    ]
    x_sentences = e.process_input(sentences)
    s_embedded = e.embed(x_sentences)

    # compute global normal vector
    global_norm = attended_norm(s_embedded)[None, :]
    top_sentences, top_scores, _ = text_pass_filter(
        texts=sentences,
        texts_embeddings=s_embedded,
        net_vector=global_norm
    )
    results = {
        "elapsed_time": time.time() - st,
        "data": [{
            "text": text,
            "relevanceScore": score
        } for text, score in zip(top_sentences, top_scores.astype(float))]
    }
    return responder(results=results, error=200, message="Success")


def run_server(port=8008, is_production=True):
    """
    This initializes the Tornado WSGI server to allow for
    asynchronous request handling
    :param port: (int)
    :param is_production: determines whether the environment is production, turns off reloading (bool, default True)
    """

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)

    parse_command_line()

    io_loop = IOLoop.instance()
    if not is_production:
        tornado.autoreload.start(check_time=500)
    print("Listening to port", port)

    try:
        io_loop.start()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run_server(port=8008, is_production=False)
