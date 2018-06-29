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
e = Embedder(use_gpu=True)


def responder(results, error, message):
    assert isinstance(results, dict)
    results["message"] = message
    results = json.dumps(results, indent=2)

    return Response(
        response=results,
        status=error,
        mimetype="application/json"
    )


def cosine_similarity_sort(x, y):
    assert x.shape[0] == 1
    assert x.shape[-1] == y.shape[-1]

    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)

    similarity = np.dot(x, y.T)
    sort = np.argsort(1 - similarity, axis=1)[0]

    return sort, similarity.flatten()[sort]


def choose(sentences, scores, embeddings):
    if scores.shape[0] == 1:
        return sentences, scores, embeddings

    # should be sigmoidal since the selections are independent variables
    prob_correct = 1 / (1 + np.exp(-(scores - scores.mean()) / scores.std()))
    threshold = prob_correct.mean()

    cut = prob_correct >= threshold
    return sentences[cut], scores[cut], embeddings[cut]


@app.route('/condense', methods=['POST', 'GET'])
@cross_origin(origins=['*'], allow_headers=['Content-Type', 'Authorization'])
def compute():
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

    # get the embedding vectors for each paragraph as groups
    paragraphs = th.split_paragraphs(body)
    x_paragraph = e.process_input(paragraphs)
    embedding_paragraph = e.embed(x_paragraph)

    # get the embedding vectors for each sentence in the document
    sentences = [th.split_sentences(sent) for sent in paragraphs]
    x_sentence = np.vstack([e.process_input([sent]) for para in sentences for sent in para])
    embedding_sentence = e.embed(x_sentence)

    # loop through each paragraph, compute the sentence that best summarizes the paragraph
    # using cosine similarity scores between each sentence embedding and overall paragraph embedding
    best_sentences, best_scores, best_embeddings = [], [], []
    start = 0
    for p in range(len(sentences)):
        num_sentences = len(sentences[p])
        s_emb = embedding_sentence[start: start + num_sentences]
        p_emb = embedding_paragraph[p, None]

        sorted_order, scores = cosine_similarity_sort(p_emb, s_emb)
        sorted_sentences = np.array(sentences[p])[sorted_order]
        embeddings = s_emb[sorted_order]

        sorted_sentences = sorted_sentences[~np.isnan(scores)]
        scores = scores[~np.isnan(scores)]

        if sorted_sentences.shape[0] > 0:
            top_sentences, top_scores, top_embeddings = choose(sorted_sentences, scores, embeddings)

            best_sentences.extend(top_sentences)
            best_scores.extend(top_scores)
            best_embeddings.extend(top_embeddings)

        start += num_sentences

    # get the embedding vector for the combination of the best sentences
    x_agg = e.process_input([" ".join(best_sentences)])
    embedding_agg = e.embed(x_agg)

    # for each "best sentence" in each paragraph, which one best summarizes the entire document
    sorted_order, scores = cosine_similarity_sort(embedding_agg, np.vstack(best_embeddings))
    ordered = np.array(best_sentences)[sorted_order]

    top_sentences, top_scores, _ = choose(ordered, scores, np.array(best_embeddings)[sorted_order])
    # impose 90% cutoff on the posterior probabilities
    top_sentences = top_sentences[top_scores > 0.9]
    top_scores = top_scores[top_scores > 0.9]

    data = [{"text": text, "relevanceScore": score} for text, score in zip(top_sentences, top_scores.astype(float))]

    results = {
        "elapsed_time": time.time() - st,
        "data": data
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
