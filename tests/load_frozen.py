import tensorflow as tf
import numpy as np

import pickle

import argparse
import os

from bin import utils

root = os.path.dirname(os.path.abspath(__file__)) + "/../../text2vec/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_name", type=str, help="Folder name in which to store model.")
    args = parser.parse_args()

    if args.model_name is None:
        print(args.print_help())
        exit(2)

    log_dir = root + args.model_name
    model_dir = log_dir + "/saved"
    name = "embedder"
    with open(log_dir + "/lookup.pkl", "rb") as pf:
        lookup = pickle.load(pf)

    test_sentences = [
        "The movie was great!",
        "The movie was terrible."
    ]
    text_x = np.array([
        utils.pad_sequence(seq, lookup.max_sequence_length)
        for seq in lookup.transform(test_sentences)
    ])

    with tf.gfile.GFile(model_dir + '/frozen_model.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def_optimized, name=name)

    with tf.Session(graph=graph) as sess:
        embed_tensor_names = [
            op.name for op in graph.get_operations()
            if 'context' in op.name and 'attention' in op.name and op.type != 'Const']

        assert len(embed_tensor_names) == 1
        seq_input = graph.get_tensor_by_name(f'{name}/sequence-input:0')
        embedding = graph.get_tensor_by_name(embed_tensor_names[0] + ':0')

        vectors = sess.run(embedding, {seq_input: text_x})
        print(vectors)
