from text2vec.preprocessing import EmbeddingLookup
from text2vec.models import TextAttention, Tensor2Tensor
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
import numpy as np

import pickle

import argparse
import json
import os

root = os.path.dirname(os.path.abspath(__file__))


def log(message):
    print(f"[INFO] {message}")


def validate_hardware():
    has_gpu = any([True if x.device_type == 'GPU' else False for x in list_local_devices()])

    if not has_gpu:
        log("Use of the GPU was requested but not found. Placing graph on the CPU.")
    return has_gpu


def load_text():
    path = root + "/../text2vec/data/"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    texts = []

    for file in files:
        with open(path + file, "r", encoding="latin1") as f:
            texts.extend(f.readlines())

    return texts


def test_val_split(corpus, val_size):
    s = np.random.permutation(range(len(corpus)))
    cv_set = [corpus[item] for item in s[:val_size]]
    corpus = [corpus[item] for item in s[val_size:]]
    return corpus, cv_set


def pad_sequence(sequence, max_sequence_length):
    """
    Pads individual text sequences to the maximum length
    seen by the model at training time
    :param sequence: list of integer lookup keys for the vocabulary (list)
    :param max_sequence_length: (int)
    :return: padded sequence (ndarray)
    """

    sequence = np.array(sequence, dtype=np.int32)
    difference = max_sequence_length - sequence.shape[0]
    pad = np.zeros((difference,), dtype=np.int32)
    return np.concatenate((sequence, pad))


def mini_batches(corpus, size, n_batches, max_len, seed):
    np.random.seed(seed)
    s = np.random.choice(range(len(corpus)), replace=False, size=min(len(corpus), size * n_batches)).astype(np.int32)

    for mb in range(n_batches):
        yield np.array([pad_sequence(corpus[index], max_len) for index in s[mb * size: (mb + 1) * size]])


def compute_angles(vectors):
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.dot(vectors, vectors.T)
    cosine = np.clip(cosine, -1, 1)
    degrees = np.arccos(cosine) * (180 / np.pi)
    return degrees


def train(model_folder, num_tokens=10000, embedding_size=256, num_hidden=128, attention_size=128, layers=8,
          batch_size=32, num_batches=50, num_epochs=10, use_tf_idf=False):

    log_dir = root + "/../../text2vec/" + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log("Fetching corpus and transforming to frequency domain")
    corpus = load_text()

    log("Splitting the training and validation sets")
    full_text, cv_x = test_val_split(corpus=corpus, val_size=512)

    log("Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens, use_tf_idf_importance=use_tf_idf)
    full_text = lookup.fit_transform(corpus=full_text)
    cv_x = lookup.transform(corpus=cv_x)

    log("Getting the maximum sequence length and vocab size")
    max_seq_len = max([len(seq) for seq in full_text + cv_x])
    vocab_size = max([max(seq) for seq in full_text + cv_x]) + 1

    with open(log_dir + "/lookup.pkl", "wb") as pf:
        pickle.dump(lookup, pf)

    # write word lookup to a TSV file for TensorBoard visualizations
    with open(log_dir + "/metadata.tsv", "w") as lf:
        reverse = lookup.reverse
        lf.write("<eos>\n")
        for k in reverse:
            lf.write(reverse[k] + '\n')

    log(f"Padding sequences in corpus to length {max_seq_len}")
    full_text = np.array([pad_sequence(seq, max_seq_len) for seq in full_text])
    cv_x = np.array([pad_sequence(seq, max_seq_len) for seq in cv_x])
    keep_probabilities = [0.9, 0.9, 1.0]

    log("Building computation graph")
    seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len])
    keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

    file_sys = open(log_dir + "/model.json", "w")
    meta_data = {
        "embeddingDimensions": num_hidden,
        "maxSequenceLength": max_seq_len,
        "vocabSize": vocab_size,
        "attentionWeightDim": attention_size,
        "trainingParameters": {
            "keepProbabilities": keep_probabilities,
            "nBatches": num_batches,
            "batchSize": batch_size,
            "maxEpochs": num_epochs
        }
    }
    json.dump(meta_data, file_sys, indent=2)
    file_sys.close()

    # model = TextAttention(
    #     input_x=seq_input,
    #     embedding_size=100,
    #     vocab_size=vocab_size,
    #     keep_prob=keep_prob,
    #     num_hidden=num_hidden,
    #     attention_size=attention_size,
    #     is_training=True,
    #     use_cuda=use_cuda
    # )

    model = Tensor2Tensor(
        input_x=seq_input,
        embedding_size=embedding_size,
        vocab_size=vocab_size,
        keep_prob=keep_prob,
        layers=layers,
        is_training=True
    )

    lstm_file_name = None
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.8,
        allow_growth=True
    )
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )
    test_sentences = [
        "The movie was great!",
        "The movie was terrible."
    ]
    text_x = np.array([pad_sequence(seq, max_seq_len) for seq in lookup.transform(test_sentences)])

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        summary_writer_train = tf.summary.FileWriter(log_dir + '/training', sess.graph)
        summary_writer_dev = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
        summary_writer_train.add_graph(graph=sess.graph)
        summary_writer_train.flush()

        # add metadata to embeddings for visualization purposes
        config_ = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_conf = config_.embeddings.add()
        embeddings = sess.graph.get_tensor_by_name("word-embeddings:0")
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = log_dir + "/metadata.tsv"
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer_train, config_)

        # model.assign_lr(sess, 1.0)
        model.assign_lr(sess, 0.01)
        model.assign_clip_norm(sess, 100000.0)

        for epoch in range(num_epochs):
            print("\t Epoch: {0}".format(epoch + 1))
            train_summary = tf.Summary()
            i = 1

            for x in mini_batches(full_text, size=batch_size, n_batches=num_batches, max_len=max_seq_len, seed=epoch):
                if x.shape[0] == 0:
                    continue

                loss_val, gradient, _ = sess.run(
                    [model.loss, model.gradient_norm, model.train],
                    feed_dict={
                        seq_input: x,
                        keep_prob: keep_probabilities
                    }
                )

                train_summary.value.add(tag="cost", simple_value=loss_val)
                train_summary.value.add(tag="gradient_norm", simple_value=gradient)

                summary_writer_train.add_summary(train_summary, epoch * num_batches + i)

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(i, loss_val))

                    # for var in tf.trainable_variables():
                    #     tf.summary.histogram(var.op.name, var)
                    # merged = sess.run(tf.summary.merge_all())
                    # summary_writer_train.add_summary(merged, epoch * num_batches + i)
                i += 1
            summary_writer_train.flush()

            vectors = sess.run(model.embedding, feed_dict={seq_input: text_x})
            angle = compute_angles(vectors)[0, 1]
            print(f"The 'angle' between `{'` and `'.join(test_sentences)}` is {angle} degrees")

            test_case_summary = tf.Summary()
            test_case_summary.value.add(tag="similarity angle", simple_value=angle)
            summary_writer_dev.add_summary(test_case_summary, epoch * num_batches + i)
            summary_writer_dev.flush()

            # dev_summary = tf.Summary()
            # cv_loss = sess.run(model.loss, feed_dict={seq_input: cv_x, keep_prob: keep_probabilities})
            # dev_summary.value.add(tag="cost", simple_value=cv_loss)
            # summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
            # summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

    return lstm_file_name


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train", "infer"], help="Run type.")
    parser.add_argument("model_name", type=str, help="Folder name in which to store model.")
    parser.add_argument("--tokens", type=int, help="Set the number of tokens to use.", default=10000)
    parser.add_argument("--embedding", type=int, help="Set the dimensionality of the word embeddings.", default=256)
    parser.add_argument("--hidden", type=int, help="Number of hidden LSTM dimensions.", default=128)
    parser.add_argument("--attention_size", type=int, help="Dimension of attention mechanism weight.", default=128)
    parser.add_argument("--layers", type=int, help="Number of self-attention layers.", default=8)
    parser.add_argument("--mb_size", type=int, help="Number of examples in each mini-batch.", default=32)
    parser.add_argument("--num_mb", type=int, help="Number of mini-batches per epoch.", default=40)
    parser.add_argument("--epochs", type=int, help="Number of epochs to run.", default=100000)
    parser.add_argument("--idf", action='store_true', help="Flag set to use TF-IDF values for N-token selection.")

    args = parser.parse_args()

    if args.run is None or args.model_name is None:
        print(args.print_help())
        exit(2)

    if args.run == "train":
        train(
            model_folder=args.model_name,
            num_tokens=args.tokens,
            embedding_size=args.embedding,
            num_hidden=args.hidden,
            attention_size=args.attention_size,
            layers=args.layers,
            batch_size=args.mb_size,
            num_batches=args.num_mb,
            num_epochs=args.epochs,
            use_tf_idf=bool(args.idf)
        )
    elif args.run == "infer":
        os.environ["MODEL_PATH"] = args.model_name
        from .text_summarize import run_server
        run_server(port=8008, is_production=False)
    else:
        raise NotImplementedError("Only training and inferencing is enabled right now.")
    return


if __name__ == '__main__':
    main()
