from text2vec.preprocessing import EmbeddingLookup
from text2vec import models
import tensorflow as tf
import numpy as np
from . import utils

import pickle

import argparse
import json
import os

root = os.path.dirname(os.path.abspath(__file__))


def load_text():
    """
    Loads the training data from a text file
    :return: (list)
    """

    path = root + "/../text2vec/data/"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    texts = []

    for file in files:
        with open(path + file, "r", encoding="latin1") as f:
            texts.extend(f.readlines())

    return texts


def test_val_split(corpus, val_size):
    """
    Splits the entire corpus into training and validation sets
    :param corpus: all training documents (list)
    :param val_size: number of examples in the validation set (int)
    :return: training set, validation set (list, list)
    """

    s = np.random.permutation(range(len(corpus)))
    cv_set = [corpus[item] for item in s[:val_size]]
    corpus = [corpus[item] for item in s[val_size:]]
    return corpus, cv_set


def mini_batches(corpus, size, n_batches, max_len, seed):
    """
    Mini-batch generator for feeding training examples into the model
    :param corpus: training set of sequence-encoded data (list)
    :param size: size of each mini-batch (int)
    :param n_batches: number of mini-batches in each epoch (int)
    :param max_len: maximum sequence (time-steps) length (int)
    :param seed: numpy randomization seed (int)
    :return: mini-batch, dimensions are [batch_size, max_len] (np.ndarray)
    """

    np.random.seed(seed)
    s = np.random.choice(range(len(corpus)), replace=False, size=min(len(corpus), size * n_batches)).astype(np.int32)

    for mb in range(n_batches):
        yield np.array([utils.pad_sequence(corpus[index], max_len) for index in s[mb * size: (mb + 1) * size]])


def train(model_folder, num_tokens=10000, embedding_size=256, num_hidden=128, attention_size=128, layers=8,
          batch_size=32, num_batches=50, num_epochs=10, glove_embeddings_file=None,
          use_tf_idf=False, use_attention=False, verbose=False):
    """
    Core training algorithm
    :param model_folder: name of the folder to create for the trained model (str)
    :param num_tokens: number of vocab tokens to keep from the training corpus,
                       is mutable if the GloVe option is chosen (int, optional)
    :param embedding_size: size of the word-embedding dimensions,
                           is overridden if the GloVe option is chosen (int, optional)
    :param num_hidden: number of hidden LSTM dimensions (int, optional)
    :param attention_size: number of hidden attention-mechanism dimensions (int, optional)
    :param layers: number of multi-head attention mechanisms for transformer model (int, optional)
    :param batch_size: size of each mini-batch (int, optional)
    :param num_batches: number of mini-batches in each epoch (int, optional)
    :param num_epochs: number of training epochs (int, optional)
    :param glove_embeddings_file: file location of the pre-trained GloVe embeddings,
                                  if set will override some settings above (str, optional)
    :param use_tf_idf: set to True to choose embedding tokens based on TF-IDF values rather than frequency alone (bool)
    :param use_attention: set to True to use the self-attention only model (bool)
    :param verbose: set to True to log learned weight distributions and validation set performance (bool, optional)
    """

    log_dir = root + "/../../text2vec/" + model_folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    utils.log("Fetching corpus and transforming to frequency domain")
    corpus = load_text()

    utils.log("Splitting the training and validation sets")
    train_corpus, cv_corpus = test_val_split(corpus=corpus, val_size=512)

    utils.log("Fitting embedding lookup and transforming the training and cross-validation sets")
    lookup = EmbeddingLookup(top_n=num_tokens, use_tf_idf_importance=use_tf_idf)
    full_text = lookup.fit_transform(corpus=train_corpus)
    cv_x = lookup.transform(corpus=cv_corpus)

    weights = None
    if glove_embeddings_file is not None:
        utils.log("Using GloVe embeddings from Common Crawl")
        (weights, unk, pad), glove_vocab = utils.load_glove_vectors(lookup, glove_path=glove_embeddings_file)
        full_text = lookup.fit_transform(corpus=train_corpus, vocab_set=set(glove_vocab))
        _, ordering = zip(*sorted([(word, lookup[word]) for word in glove_vocab], key=lambda z: z[1]))
        ordering = np.array(ordering, np.int32)
        weights = np.vstack([weights, unk])[ordering - 1]  # re-order the weights to match this particular vocabulary
        weights = np.vstack([pad, weights])
        cv_x = lookup.transform(corpus=cv_corpus)
        del(unk, pad, ordering, glove_vocab)

    utils.log("Getting the maximum sequence length and vocab size")
    vocab_size = max([max(seq) for seq in full_text + cv_x]) + 1

    with open(log_dir + "/lookup.pkl", "wb") as pf:
        pickle.dump(lookup, pf)

    # write word lookup to a TSV file for TensorBoard visualizations
    with open(log_dir + "/metadata.tsv", "w") as lf:
        reverse = lookup.reverse
        for k in reverse:
            lf.write(reverse[k] + '\n')

    utils.log(f"Padding sequences in corpus to length {lookup.max_sequence_length}")
    full_text = np.array([utils.pad_sequence(seq, lookup.max_sequence_length) for seq in full_text])
    cv_x = np.array([utils.pad_sequence(seq, lookup.max_sequence_length) for seq in cv_x])
    keep_probabilities = [0.9, 0.9, 1.0]

    utils.log("Building computation graph")
    file_sys = open(log_dir + "/model.json", "w")
    meta_data = {
        "embeddingDimensions": num_hidden,
        "maxSequenceLength": lookup.max_sequence_length,
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

    if use_attention:
        model = models.Tensor2Tensor(
            max_sequence_len=lookup.max_sequence_length,
            embedding_size=embedding_size,
            vocab_size=vocab_size,
            word_weights=weights,
            layers=layers,
            is_training=True
        )
    else:
        model = models.TextAttention(
            max_sequence_len=lookup.max_sequence_length,
            embedding_size=embedding_size,
            vocab_size=vocab_size,
            num_hidden=num_hidden,
            attention_size=attention_size,
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
    text_x = np.array([utils.pad_sequence(seq, lookup.max_sequence_length) for seq in lookup.transform(test_sentences)])

    # builder = tf.saved_model.builder.SavedModelBuilder(export_dir=log_dir + "/saved")
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
        if use_attention:
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        else:
            embeddings = sess.graph.get_tensor_by_name("embedding/embeddings:0")
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = log_dir + "/metadata.tsv"
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer_train, config_)

        model.assign_clip_norm(sess, 100000.0)

        step = 1
        warm_up_steps = 4000
        for epoch in range(num_epochs):
            print("\t Epoch: {0}".format(epoch + 1))
            train_summary = tf.Summary()
            i = 1

            for x in mini_batches(full_text, size=batch_size, n_batches=num_batches,
                                  max_len=lookup.max_sequence_length, seed=epoch):
                if x.shape[0] == 0:
                    continue

                learning_rate = embedding_size ** (-0.5) * min(step ** (-0.5), step * warm_up_steps ** (-1.5))
                model.assign_lr(sess, learning_rate)
                loss_val, gradient, current_lr, _ = sess.run(
                    [model.loss, model.gradient_norm, model.lr, model.train],
                    feed_dict={
                        model.seq_input: x,
                        model.keep_prob: keep_probabilities
                    }
                )

                train_summary.value.add(tag="cost", simple_value=loss_val)
                train_summary.value.add(tag="gradient_norm", simple_value=gradient)
                train_summary.value.add(tag="learning_rate", simple_value=current_lr)

                summary_writer_train.add_summary(train_summary, epoch * num_batches + i)

                if i % (num_batches // 10) == 0:
                    print("\t\t iteration {0} - loss: {1}".format(i, loss_val))

                    if verbose:
                        for var in tf.trainable_variables():
                            tf.summary.histogram(var.op.name, var)
                        merged = sess.run(tf.summary.merge_all())
                        summary_writer_train.add_summary(merged, epoch * num_batches + i)
                i += 1
                step += 1
            summary_writer_train.flush()

            vectors = sess.run(model.embedding, feed_dict={model.seq_input: text_x})
            angle = utils.compute_angles(vectors)[0, 1]
            print(f"The 'angle' between `{'` and `'.join(test_sentences)}` is {angle} degrees")

            test_case_summary = tf.Summary()
            test_case_summary.value.add(tag="similarity angle", simple_value=angle)
            summary_writer_dev.add_summary(test_case_summary, epoch * num_batches + i)
            summary_writer_dev.flush()

            if verbose:
                dev_summary = tf.Summary()
                cv_loss = sess.run(model.loss, feed_dict={model.seq_input: cv_x, model.keep_prob: keep_probabilities})
                dev_summary.value.add(tag="cost", simple_value=cv_loss)
                summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
                summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

        tf.saved_model.simple_save(
            session=sess,
            export_dir=log_dir + "/saved",
            inputs={'sequences': model.seq_input},
            outputs={'embedding': model.embedding}
        )
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
    parser.add_argument("--attention", action='store_true', help="Set to use attention transformer model.")
    parser.add_argument("--use_glove", action='store_true', help="Set to use the GloVe Common Crawl embeddings.")
    parser.add_argument("--glove_file", type=str, help="GloVe embeddings file.", default=None)

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
            glove_embeddings_file=args.glove_file,
            use_tf_idf=bool(args.idf),
            use_attention=bool(args.attention)
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
