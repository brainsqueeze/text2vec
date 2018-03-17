from text2vec.preprocessing import EmbeddingLookup
from text2vec.models import TextAttention

import tensorflow as tf
import numpy as np

import argparse
import os

root = os.path.dirname(os.path.abspath(__file__))


def _transform_text(top_n=10000):
    path = root + "/../text2vec/data/"
    lookup = EmbeddingLookup(path=path)

    corpus = lookup.token_replace_id(top_n=top_n)
    return corpus


def _pad_sequence(sequence, max_sequence_length):
    seq_len = len(sequence)
    difference = max_sequence_length - seq_len
    pad = [0] * difference
    return np.array(sequence + pad)


def mini_batches(corpus, size, n_batches, seed):
    np.random.seed(seed)
    s = np.random.permutation(range(len(corpus)))

    return [np.array([corpus[elem] for elem in s[n * size: (n + 1) * size]]) for n in range(n_batches)]


def train(model_folder, num_tokens=10000, num_hidden=128, attention_size=128, num_epochs=10):
    print("[INFO] Fetching corpus and transforming to frequency domain")
    corpus = _transform_text(top_n=num_tokens)
    max_seq_len = max([len(seq) for seq in corpus])
    vocab_size = max([max(seq) for seq in corpus]) + 1

    print("[INFO] Padding sequences in corpus to length", max_seq_len)
    corpus = np.array([_pad_sequence(seq, max_seq_len) for seq in corpus])
    keep_probabilities = [0.5, 0.8, 0.6]
    num_batches = 60

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
    log_dir = root + "/../text2vec/" + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        summary_writer_train = tf.summary.FileWriter(log_dir + '/training', sess.graph)
        # summary_writer_dev = tf.summary.FileWriter(log_dir + '/dev', sess.graph)
        summary_writer_train.add_graph(graph=sess.graph)
        summary_writer_train.flush()

        model.assign_lr(sess, 1.0)
        model.assign_clip_norm(sess, 10.0)
        for epoch in range(num_epochs):
            print("\t Epoch: %d" % (epoch + 1))
            i = 1

            for x in mini_batches(corpus, size=16, n_batches=num_batches, seed=epoch):
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

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

    return lstm_file_name


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train"], help="Run type.")
    parser.add_argument("--tokens", type=int, help="Set the number of tokens to use, defaults to 10,000.")
    parser.add_argument("--hidden", type=int, help="Number of hidden LSTM dimensions.")
    parser.add_argument("--attention_size", type=int, help="Dimension of attention mechanism weight.")
    parser.add_argument("--epochs", type=int, help="Number of epochs to run.")
    parser.add_argument("--model_name", type=str, help="Folder name in which to store model.")

    args = parser.parse_args()

    if args.run == "train":
        train(
            model_folder=args.model_name,
            num_tokens=args.tokens,
            num_hidden=args.hidden,
            attention_size=args.attention_size,
            num_epochs=args.epochs
        )
    else:
        raise NotImplementedError("Only training is enabled right now.")
    return


if __name__ == '__main__':
    main()
