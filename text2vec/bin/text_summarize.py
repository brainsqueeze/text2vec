from .serving_tools import Embedder
import numpy as np

import time
import json

from flask import Flask, request, Response
from flask_cors import cross_origin

from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
import tornado.autoreload
import tornado

app = Flask(__name__)
model = Embedder()


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

    similarity = np.dot(embedding_matrix, net_vector)
    similarity = np.clip(similarity, -1, 1)
    # sort = np.argsort(1 - similarity)
    sort = np.argsort(similarity - 1)

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
    # likelihood = np.exp(-angles) / np.exp(-angles).sum()
    # threshold = likelihood.mean() + likelihood.std()
    # cut = likelihood >= threshold
    # cut = angles < 45
    cut = angles < angles.mean()
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
    sentences, vectors, doc_vector = model.embed(body)
    top_sentences, top_scores, _ = text_pass_filter(texts=sentences, texts_embeddings=vectors, net_vector=doc_vector)

    results = {
        "elapsed_time": time.time() - st,
        "data": [{
            "text": text,
            "relevanceScore": score
        } for text, score in zip(top_sentences, top_scores.astype(float))]
    }
    return responder(results=results, error=200, message="Success")


def run_server(port=8008):
    """
    This initializes the Tornado WSGI server to allow for
    asynchronous request handling
    :param port: (int)
    """

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)

    io_loop = IOLoop.instance()
    tornado.autoreload.start(check_time=500)
    print("Listening to port", port)

    try:
        io_loop.start()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run_server(port=8008)
