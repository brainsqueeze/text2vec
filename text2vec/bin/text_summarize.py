import time
import json

from flask import Flask, request, Response
from flask_cors import cross_origin

from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
import tornado.autoreload
import tornado

import numpy as np
from .serving_tools import Embedder

app = Flask(__name__)
model = Embedder()


def responder(results, error, message):
    """Boilerplate Flask response item.

    Parameters
    ----------
    results : dict
        API response
    error : int
        Error code
    message : str
        Message to send to the client

    Returns
    -------
    flask.Reponse
    """

    assert isinstance(results, dict)
    results["message"] = message
    results = json.dumps(results, indent=2)

    return Response(
        response=results,
        status=error,
        mimetype="application/json"
    )


def cosine_similarity_sort(net_vector, embedding_matrix):
    """
    Computes the cosine similarity scores and then returns
    the sorted results

    Parameters
    ----------
    net_vector : np.ndarray
        The context vector for the entire document
    embedding_matrix : np.ndarray
        The context vectors (row vectors) for each constituent body of text

    Returns
    -------
    (ndarray, ndarray)
        (sorted order of documents, cosine similarity scores)
    """

    similarity = np.dot(embedding_matrix, net_vector)
    similarity = np.clip(similarity, -1, 1)
    # sort = np.argsort(1 - similarity)
    sort = np.argsort(similarity - 1)

    return sort, similarity.flatten()[sort]


def angle_from_cosine(cosine_similarity):
    """
    Computes the angles in degrees from cosine similarity scores

    Parameters
    ----------
    cosine_similarity : np.ndarray

    Returns
    -------
    ndarray
        Cosine angles (num_sentences,)
    """

    return np.arccos(cosine_similarity) * (180 / np.pi)


def choose(sentences, scores, embeddings):
    """
    Selects the best constituent texts from the similarity scores

    Parameters
    ----------
    sentences : np.ndarray
        Array of the input texts, sorted by scores.
    scores : np.ndarray
        Cosine similarity scores, sorted
    embeddings : np.ndarray
        Embedding matrix for input texts, sorted by scores

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        (best sentences sorted, best scores sorted, best embeddings sorted)
    """

    if scores.shape[0] == 1:
        return sentences, scores, embeddings

    angles = angle_from_cosine(scores)
    cut = angles < angles.mean() - angles.std()
    return sentences[cut], scores[cut], embeddings[cut]


def text_pass_filter(texts, texts_embeddings, net_vector):
    """
    Runs the scoring + filtering process on input texts

    Parameters
    ----------
    texts : np.ndarray
        Input texts.
    texts_embeddings : np.ndarray
        Context embedding matrix for input texts.
    net_vector : np.ndarray
        The context vector for the entire document

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        (best sentences sorted, best scores sorted, best embeddings sorted)
    """

    sorted_order, scores = cosine_similarity_sort(net_vector=net_vector, embedding_matrix=texts_embeddings)
    texts = np.array(texts)[sorted_order]
    filtered_texts, filtered_scores, filtered_embeddings = choose(
        sentences=texts,
        scores=scores,
        embeddings=texts_embeddings[sorted_order]
    )

    return filtered_texts, filtered_scores, filtered_embeddings


def softmax(logits):
    """
    Computes the softmax of the input logits.

    Parameters
    ----------
    logits : np.ndarray

    Returns
    -------
    np.ndarray
        Softmax output array with the same shape as the input.
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

    Returns
    -------
    flask.Response
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
    """This initializes the Tornad WSGI server to allow robust request handling.

    Parameters
    ----------
    port : int, optional
        Port number to serve the app on, by default 8008
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
