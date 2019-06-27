from text2vec.preprocessing import utils as str_utils
from text2vec import models
import tensorflow as tf
import numpy as np
from . import utils

from tensorboard.plugins import projector

import argparse
import os

root = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_text(data_path=None, max_length=-1):
    """
    Loads the training data from a text file
    :return: (list)
    """

    path = f"{data_path}/" if data_path else f"{root}/../text2vec/data/"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    texts = []

    for file in files:
        with open(path + file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line != '':
                    if max_length > 0:
                        if len(line.split()) <= max_length:
                            texts.append(line)
                    else:
                        texts.append(line)

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


def mini_batches(corpus, size, n_batches, seed):
    """
    Mini-batch generator for feeding training examples into the model
    :param corpus: training set of sequence-encoded data (list)
    :param size: size of each mini-batch (int)
    :param n_batches: number of mini-batches in each epoch (int)
    :param seed: numpy randomization seed (int)
    :return: mini-batch, dimensions are [batch_size, max_len] (np.ndarray)
    """

    np.random.seed(seed)
    s = np.random.choice(range(len(corpus)), replace=False, size=min(len(corpus), size * n_batches)).astype(np.int32)

    for mb in range(n_batches):
        yield [' '.join(str_utils.clean_and_split(corpus[index])) for index in s[mb * size: (mb + 1) * size]]


def train(model_folder, num_tokens=10000, embedding_size=256, num_hidden=128, max_allowed_seq=-1,
          attention_size=128, layers=8, batch_size=32, num_batches=50, num_epochs=10,
          glove_embeddings_file=None, data_path=None, model_path=None,
          use_tf_idf=False, use_attention=False, verbose=False):
    """
    Core training algorithm
    :param model_folder: name of the folder to create for the trained model (str)
    :param num_tokens: number of vocab tokens to keep from the training corpus,
                       is mutable if the GloVe option is chosen (int, optional)
    :param embedding_size: size of the word-embedding dimensions,
                           is overridden if the GloVe option is chosen (int, optional)
    :param num_hidden: number of hidden LSTM dimensions (int, optional)
    :param max_allowed_seq: the maximum sequence length allowed, model will truncate if longer (int)
    :param attention_size: number of hidden attention-mechanism dimensions (int, optional)
    :param layers: number of multi-head attention mechanisms for transformer model (int, optional)
    :param batch_size: size of each mini-batch (int, optional)
    :param num_batches: number of mini-batches in each epoch (int, optional)
    :param num_epochs: number of training epochs (int, optional)
    :param glove_embeddings_file: file location of the pre-trained GloVe embeddings,
                                  if set will override some settings above (str, optional)
    :param data_path: valid path to the training data (str)
    :param model_path: valid path to where the model will be saved (str)
    :param use_tf_idf: set to True to choose embedding tokens based on TF-IDF values rather than frequency alone (bool)
    :param use_attention: set to True to use the self-attention only model (bool)
    :param verbose: set to True to log learned weight distributions and validation set performance (bool, optional)
    """

    log_dir = f"{model_path}/{model_folder}" if model_path else f"{root}/../../text2vec/{model_folder}"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    utils.log("Fetching corpus and transforming to frequency domain")
    corpus = load_text(data_path=data_path, max_length=max_allowed_seq)

    utils.log("Splitting the training and validation sets")
    train_corpus, cv_corpus = test_val_split(corpus=corpus, val_size=64)
    cv_tokens = [' '.join(str_utils.clean_and_split(text)) for text in cv_corpus]

    utils.log("Fitting embedding lookup and transforming the training and cross-validation sets")
    hash_map, max_sequence_length = str_utils.get_top_tokens(corpus, n_top=num_tokens)
    utils.log(f"Max sequence length: {max_sequence_length}")

    with open(log_dir + "/metadata.tsv", "w") as tsv:
        for token, _ in sorted(hash_map.items(), key=lambda s: s[-1]):
            tsv.write(token + "\n")
        tsv.write("<unk>\n")

    # weights = None
    if glove_embeddings_file is not None:
        raise NotImplementedError("GloVe embeddings not supported at this time")
    #     utils.log("Using GloVe embeddings from Common Crawl")
    #     (weights, unk, eos, bos), glove_vocab = utils.load_glove_vectors(lookup, glove_path=glove_embeddings_file)
    #     glove_vocab.append(lookup.unknown)  # add the unknown sequence tag to the GloVe vocab
    #     full_text = lookup.fit_transform(corpus=train_corpus, vocab_set=set(glove_vocab))
    #     _, ordering = zip(*sorted([(word, lookup[word]) for word in glove_vocab], key=lambda z: z[1]))
    #     ordering = np.array(ordering, np.int32)
    #     weights = np.vstack([weights, unk])[ordering - 1]  # re-order the weights to match this particular vocabulary
    #     weights = np.vstack([eos, bos, weights])
    #     cv_x = lookup.transform(corpus=cv_corpus)
    #     del(unk, eos, bos, ordering, glove_vocab)

    keep_probabilities = [0.9, 0.75, 1.0]

    utils.log("Building computation graph")
    if use_attention:
        model = models.Transformer(
            max_sequence_len=max_sequence_length,
            embedding_size=embedding_size,
            token_hash=hash_map,
            layers=layers,
            n_stacks=1
        )
    else:
        raise NotImplementedError("Only the Transformer model is currently available")
        # model = models.TextAttention(
        #     max_sequence_len=lookup.max_sequence_length,
        #     embedding_size=embedding_size,
        #     vocab_size=vocab_size,
        #     num_hidden=num_hidden,
        #     attention_size=attention_size,
        #     is_training=True
        # )

    lstm_file_name = None
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
    test_tokens = [' '.join(str_utils.clean_and_split(text)) for text in test_sentences]

    with tf.Session(config=sess_config) as sess:
        saver = tf.train.Saver()
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        summary_writer_train = tf.summary.FileWriter(log_dir + '/training', sess.graph)
        summary_writer_dev = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
        summary_writer_train.add_graph(graph=sess.graph)
        summary_writer_train.flush()

        # add word labels to the projector
        config_ = projector.ProjectorConfig()
        embeddings_config = config_.embeddings.add()
        embeddings = model.embeddings
        embeddings_config.tensor_name = embeddings.name
        embeddings_config.metadata_path = log_dir + "/metadata.tsv"
        projector.visualize_embeddings(summary_writer_train, config_)

        model.assign_clip_norm(sess, 100000.0)

        step = 1
        warm_up_steps = 4000
        for epoch in range(num_epochs):
            print(f"\t Epoch: {epoch + 1}")
            i = 1
            summary_count = 1

            for x in mini_batches(train_corpus, size=batch_size, n_batches=num_batches, seed=epoch):
                if len(x) == 0:
                    continue

                learning_rate = embedding_size ** (-0.5) * min(step ** (-0.5), step * warm_up_steps ** (-1.5))
                model.assign_lr(sess, learning_rate)
                feed_dict = {model.enc_tokens: x, model.keep_prob: keep_probabilities}
                if i % (num_batches // 10) == 0:
                    train_summary = tf.Summary()
                    operations = [model.loss, model.gradient_norm, model.lr, model.merged, model.train]

                    if summary_count == 10:
                        run_metadata = tf.RunMetadata()
                        loss_val, gradient, current_lr, summary, _ = sess.run(
                            operations,
                            feed_dict=feed_dict,
                            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                            run_metadata=run_metadata
                        )
                        summary_writer_train.add_run_metadata(run_metadata, tag=f'{step}')
                    else:
                        loss_val, gradient, current_lr, summary, _ = sess.run(operations, feed_dict=feed_dict)

                    print(f"\t\t iteration {i} - loss: {loss_val}")
                    train_summary.value.add(tag="cost", simple_value=loss_val)
                    train_summary.value.add(tag="gradient_norm", simple_value=gradient)
                    train_summary.value.add(tag="learning_rate", simple_value=current_lr)

                    summary_writer_train.add_summary(train_summary, step)
                    summary_writer_train.add_summary(summary, step)
                    summary_writer_train.flush()
                    summary_count += 1
                else:
                    sess.run(model.train, feed_dict=feed_dict)

                i += 1
                step += 1

            if verbose:
                dev_summary = tf.Summary()
                test_case_summary = tf.Summary()

                vectors = sess.run(model.embedding, feed_dict={model.enc_tokens: test_tokens})
                angle = utils.compute_angles(vectors)[0, 1]
                print(f"The 'angle' between `{'` and `'.join(test_sentences)}` is {angle} degrees")

                test_case_summary.value.add(tag="similarity angle", simple_value=angle)
                summary_writer_dev.add_summary(test_case_summary, step)
                summary_writer_dev.flush()

                cv_loss = sess.run(
                    model.loss,
                    feed_dict={model.enc_tokens: cv_tokens, model.keep_prob: keep_probabilities}
                )
                dev_summary.value.add(tag="cost", simple_value=cv_loss)
                summary_writer_dev.add_summary(dev_summary, epoch * num_batches + i)
                summary_writer_dev.flush()

            lstm_file_name = saver.save(sess, log_dir + '/embedding_model', global_step=int((epoch + 1) * i))

        tf.compat.v1.saved_model.simple_save(
            session=sess,
            export_dir=log_dir + "/saved",
            inputs={'sequences': model.enc_tokens},
            outputs={'embedding': model.embedding},
            legacy_init_op=tf.tables_initializer()
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
    parser.add_argument("--max_len", type=int, help="Maximum allowed sequence length", default=-1)
    parser.add_argument("--use_glove", action='store_true', help="Set to use the GloVe Common Crawl embeddings.")
    parser.add_argument("--glove_file", type=str, help="GloVe embeddings file.", default=None)
    parser.add_argument("--data_path", type=str, help="Path to the training data file(s).")
    parser.add_argument("--model_path", type=str, help="Path to place the saved model.")
    parser.add_argument("--verbose", action='store_true', help="Set to evaluate the CV set after each epoch.")

    args = parser.parse_args()

    if args.run is None or args.model_name is None:
        print(args.print_help())
        exit(2)

    if args.data_path and not os.path.isdir(args.data_path):
        print(f"{args.data_path} is not a valid directory")
        exit(2)

    if args.model_path and not os.path.isdir(args.model_path):
        print(f"{args.model_path} is not a valid directory")
        exit(2)

    if args.run == "train":
        train(
            model_folder=args.model_name,
            num_tokens=args.tokens,
            max_allowed_seq=args.max_len,
            embedding_size=args.embedding,
            num_hidden=args.hidden,
            attention_size=args.attention_size,
            layers=args.layers,
            batch_size=args.mb_size,
            num_batches=args.num_mb,
            num_epochs=args.epochs,
            glove_embeddings_file=args.glove_file,
            use_tf_idf=bool(args.idf),
            use_attention=bool(args.attention),
            data_path=args.data_path,
            model_path=args.model_path,
            verbose=args.verbose
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
