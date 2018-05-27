from text2vec.preprocessing import EmbeddingLookup
from text2vec.models import TextAttention
import tensorflow as tf
import numpy as np

import pickle

import argparse
import os

root = os.path.dirname(os.path.abspath(__file__))


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


def _pad_sequence(sequence, max_sequence_length):
    seq_len = len(sequence)
    difference = max_sequence_length - seq_len
    pad = [0] * difference
    return np.array(sequence + pad)


def mini_batches(corpus, size, n_batches, seed):
    np.random.seed(seed)
    s = np.random.permutation(range(len(corpus)))

    return [np.array([corpus[elem] for elem in s[n * size: (n + 1) * size]]) for n in range(n_batches)]


def train(model_folder, num_tokens=10000, num_hidden=128, attention_size=128,
          batch_size=32, num_batches=50, num_epochs=10):
    log_dir = root + "/../../text2vec/" + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    print("[INFO] Fetching corpus and transforming to frequency domain")
    corpus = load_text()

    print("[INFO] Splitting the training and validation sets")
    full_text, cv_x = test_val_split(corpus=corpus, val_size=512)

    print("[INFO] Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens)
    full_text = lookup.fit_transform(corpus=full_text)
    cv_x = lookup.transform(corpus=cv_x)

    print("[INFO] Getting the maximum sequence length and vocab size")
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

    print("[INFO] Padding sequences in corpus to length", max_seq_len)
    full_text = np.array([_pad_sequence(seq, max_seq_len) for seq in full_text])
    cv_x = np.array([_pad_sequence(seq, max_seq_len) for seq in cv_x])
    keep_probabilities = [0.5, 0.8, 0.6]

    print("[INFO] Compiling seq2seq automorphism model")
    seq_input = tf.placeholder(dtype=tf.int32, shape=[None, max_seq_len])
    keep_prob = tf.placeholder_with_default([1.0, 1.0, 1.0], shape=(3,))

    model = TextAttention(
        input_x=seq_input,
        embedding_size=100,
        vocab_size=vocab_size,
        keep_prob=keep_prob,
        num_hidden=num_hidden,
        attention_size=attention_size,
        is_training=True
    )

    lstm_file_name = None
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False
    )

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
        embeddings = sess.graph.get_tensor_by_name("embedding/word_embeddings:0")
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = log_dir + "/metadata.tsv"
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer_train, config_)

        model.assign_lr(sess, 1.0)
        model.assign_clip_norm(sess, 10.0)

        for epoch in range(num_epochs):
            print("\t Epoch: %d" % (epoch + 1))
            i = 1

            for x in mini_batches(full_text, size=batch_size, n_batches=num_batches, seed=epoch):
                if x.shape[0] == 0:
                    continue
                train_summary, dev_summary = tf.Summary(), tf.Summary()

                loss_val, gradient, _ = sess.run(
                    [model.loss, model.gradient_norm, model.train],
                    feed_dict={
                        seq_input: x,
                        keep_prob: keep_probabilities
                    }
                )

                train_summary.value.add(tag="cost", simple_value=loss_val)
                train_summary.value.add(tag="gradient_norm", simple_value=gradient)

                # for var in tf.trainable_variables():
                #     tf.summary.histogram(var.op.name, var)
                # merged = sess.run(tf.summary.merge_all())
                # summary_writer_train.add_summary(merged, epoch * num_batches + i)

                summary_writer_train.add_summary(train_summary, epoch * num_batches + i)
                summary_writer_train.flush()

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(str(i), str(loss_val)))
                i += 1

            dev_summary = tf.Summary()
            cv_loss = sess.run(model.loss, feed_dict={seq_input: cv_x, keep_prob: keep_probabilities})
            dev_summary.value.add(tag="cost", simple_value=cv_loss)
            summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
            summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

    return lstm_file_name


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train"], help="Run type.")
    parser.add_argument("model_name", type=str, help="Folder name in which to store model.")
    parser.add_argument("--tokens", type=int, help="Set the number of tokens to use.", default=10000)
    parser.add_argument("--hidden", type=int, help="Number of hidden LSTM dimensions.", default=128)
    parser.add_argument("--attention_size", type=int, help="Dimension of attention mechanism weight.", default=128)
    parser.add_argument("--mb_size", type=int, help="Number of examples in each mini-batch.", default=32)
    parser.add_argument("--num_mb", type=int, help="Number of mini-batches per epoch.", default=40)
    parser.add_argument("--epochs", type=int, help="Number of epochs to run.", default=100000)

    args = parser.parse_args()

    if args.run is None or args.model_name is None:
        print(args.print_help())
        exit(2)

    if args.run == "train":
        train(
            model_folder=args.model_name,
            num_tokens=args.tokens,
            num_hidden=args.hidden,
            attention_size=args.attention_size,
            batch_size=args.mb_size,
            num_batches=args.num_mb,
            num_epochs=args.epochs
        )
    else:
        raise NotImplementedError("Only training is enabled right now.")
    return


if __name__ == '__main__':
    main()
