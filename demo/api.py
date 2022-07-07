from typing import List, Union
from math import pi
import argparse
import json
import re

from flask import Flask, request, Response, send_from_directory
from flask_cors import cross_origin
from tornado.log import enable_pretty_logging
from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
import tornado.autoreload
# from tornado import web
import tornado

import tensorflow as tf
from tensorflow.keras import models, Model
from tokenizers import Tokenizer

app = Flask(__name__, static_url_path="", static_folder="./")
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, help="Directory containing serialized model and tokenizer", required=True)
args = parser.parse_args()

model: Model = models.load_model(f"{args.model_dir}/saved_model")
tokenizer: Tokenizer = Tokenizer.from_file(f"{args.model_dir}/tokenizer.json")


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


def tokenize(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, str):
        return [' '.join(tokenizer.encode(text).tokens)]
    return [' '.join(batch.tokens) for batch in tokenizer.encode_batch(text)]


def get_summaries(paragraphs: List[str]):
    context = tf.concat([
        model.embed(batch)["attention"]
        for batch in tf.data.Dataset.from_tensor_slices(paragraphs).batch(32)
    ], axis=0)
    doc_vector = model.embed(tf.strings.reduce_join(paragraphs, separator=' ', keepdims=True))["attention"]
    cosine = tf.tensordot(tf.math.l2_normalize(context, axis=1), tf.math.l2_normalize(doc_vector, axis=1), axes=[-1, 1])
    cosine = tf.clip_by_value(cosine, -1, 1)
    likelihoods = tf.nn.softmax(180 - tf.math.acos(cosine) * (180 / pi), axis=0)
    return likelihoods


@app.route("/")
def root():
    return send_from_directory(directory="./html/", path="index.html")


@app.route("/summarize", methods=["GET", "POST"])
# @cross_origin(origins=['*'], allow_headers=['Content-Type', 'Authorization'])
def summarize():
    if request.is_json:
        payload = request.json
    else:
        payload = request.values

    text = payload.get("text", "")
    if not text:
        return responder(results={}, error=400, message="No text provided")

    paragraphs = [p for p in re.split(r"\n{1,}", text) if p.strip()]
    if len(paragraphs) < 2:
        return responder(results={"text": paragraphs}, error=400, message="Insufficient amount of text provided")

    tokenized = tokenize(paragraphs)
    likelihoods = get_summaries(tokenized)
    likelihoods = tf.squeeze(likelihoods)
    cond = tf.where(likelihoods > tf.math.reduce_mean(likelihoods) + tf.math.reduce_std(likelihoods)).numpy().flatten()
    output = [{
        "text": paragraphs[idx],
        "score": float(likelihoods[idx])
    } for idx in cond]

    results = {"data": output}
    return responder(results=results, error=200, message="Success")


def serve(port: int = 9090, debug: bool = False):
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    enable_pretty_logging()

    io_loop = IOLoop.current()
    if debug:
        tornado.autoreload.start(check_time=500)
    print("Listening to port", port, flush=True)

    try:
        io_loop.start()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    serve()
