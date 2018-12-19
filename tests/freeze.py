import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib

import argparse
import os

root = os.path.dirname(os.path.abspath(__file__)) + "/../../text2vec/"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_name", type=str, help="Folder name in which to store model.")
    args = parser.parse_args()

    if args.model_name is None:
        print(args.print_help())
        exit(2)

    model_dir = root + args.model_name + "/saved"

    with tf.Session(graph=tf.Graph()) as sess:
        model = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=model_dir)

        def_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        input_node_names = [model.signature_def[def_key].inputs['sequences'].name.split(':')[0]]
        output_node_names = [model.signature_def[def_key].outputs['embedding'].name.split(':')[0]]

        input_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=output_node_names
        )

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        tf.graph_util.remove_training_nodes(input_graph_def),
        input_node_names=input_node_names,
        output_node_names=output_node_names,
        placeholder_type_enum=tf.int32.as_datatype_enum
    )

    with tf.gfile.GFile(model_dir + '/frozen_model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
