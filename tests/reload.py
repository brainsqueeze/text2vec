import tensorflow as tf
import numpy as np

import argparse
import pickle
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
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8,
        allow_growth=True
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )

    with open(log_dir + "/lookup.pkl", "rb") as pf:
        lookup = pickle.load(pf)

    with tf.Session(config=sess_config) as sess:
        model = tf.saved_model.loader.load(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            export_dir=log_dir + "/saved"
        )

        inputs = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs
        outputs = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs

        graph = tf.get_default_graph()
        seq_input = graph.get_tensor_by_name(inputs['seq_input'].name)
        embedding = graph.get_tensor_by_name(outputs['embedding'].name)

        test_sentences = [
            "The movie was great!",
            "The movie was terrible."
        ]
        text_x = np.array([
            utils.pad_sequence(seq, lookup.max_sequence_length)
            for seq in lookup.transform(test_sentences)
        ])

        vectors = sess.run(embedding, {seq_input: text_x})
        print(vectors)
