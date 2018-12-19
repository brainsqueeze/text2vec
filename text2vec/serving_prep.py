import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib


def freeze_graph(model_dir):

    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir=model_dir)

        inputs = model.signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs
        variables = tf.get_default_graph().get_collection('trainable_variables')
        output_nodes = [var.name.split(':')[0] for var in variables if 'bahdanau' in var.name]

        input_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=tf.get_default_graph().as_graph_def(),
            output_node_names=output_nodes
        )

        output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            tf.graph_util.remove_training_nodes(input_graph_def),
            input_node_names=[inputs['seq_input'].name],
            output_node_names=output_nodes,
            placeholder_type_enum=tf.float32.as_datatype_enum
        )

        with tf.gfile.GFile('frozen_model.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
