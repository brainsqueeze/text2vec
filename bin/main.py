from text2vec.preprocessing import utils as str_utils
from text2vec.optimizer_tools import RampUpDecaySchedule
from text2vec.models import TextInput
from text2vec.models import TransformerEncoder, TransformerDecoder, RecurrentEncoder, RecurrentDecoder
from text2vec.training_tools import EncodingModel, sequence_cost, vector_cost
from . import utils

import tensorflow as tf
import numpy as np

from tensorboard.plugins import projector

from random import shuffle
import itertools
import argparse
import os

root = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_text(data_path=None, max_length=-1):
    """
    Loads the training data from a text file
    :return: (list)
    """

    path = f"{data_path}/" if data_path is not None else f"{root}/../text2vec/data/"
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    texts = []

    for file in files:
        with open(path + file, "r", encoding="utf8", errors="replace") as f:
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


def mini_batches(corpus, size):
    """
    Mini-batch generator for feeding training examples into the model
    :param corpus: training set of sequence-encoded data (list)
    :param size: size of each mini-batch (int)
    :return: mini-batch of text data (list of strings)
    """

    shuffle(corpus)

    for i in range(0, len(corpus) + size, size):
        mini_batch = corpus[i: i + size]
        if not mini_batch:
            continue
        yield [' '.join(str_utils.clean_and_split(text)) for text in mini_batch]


def train(model_folder, num_tokens=10000, embedding_size=256, num_hidden=128, max_allowed_seq=-1,
          layers=8, batch_size=32, num_epochs=10, data_path=None, model_path=None, use_attention=False,
          orthogonal_cost=False):
    """
    Core training algorithm
    :param model_folder: name of the folder to create for the trained model (str)
    :param num_tokens: number of vocab tokens to keep from the training corpus (int, optional)
    :param embedding_size: size of the word-embedding dimensions (int, optional)
    :param num_hidden: number of hidden model dimensions (int, optional)
    :param max_allowed_seq: the maximum sequence length allowed, model will truncate if longer (int)
    :param layers: number of multi-head attention mechanisms for transformer model (int, optional)
    :param batch_size: size of each mini-batch (int, optional)
    :param num_epochs: number of training epochs (int, optional)
    :param data_path: valid path to the training data (str)
    :param model_path: valid path to where the model will be saved (str)
    :param use_attention: set to True to use the self-attention only model (bool)
    :param orthogonal: set to True to add a cost to mutually parallel context vector (bool)
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
    hash_map, max_seq_len = str_utils.get_top_tokens(corpus, n_top=num_tokens)
    utils.log(f"Max sequence length: {max_seq_len}")

    with open(log_dir + "/metadata.tsv", "w") as tsv:
        for token, _ in sorted(hash_map.items(), key=lambda s: s[-1]):
            tsv.write(token + "\n")
        tsv.write("<unk>\n")

    utils.log("Building computation graph")
    log_step = len(train_corpus[::batch_size]) // 10
    dims = embedding_size

    # GPU config
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(True)

    params = dict(
        max_sequence_len=max_seq_len,
        embedding_size=dims,
        input_keep_prob=0.9,
        hidden_keep_prob=0.75
    )
    if use_attention:
        model = EncodingModel(token_hash=hash_map, layers=layers, **params)
    else:
        model = EncodingModel(token_hash=hash_map, num_hidden=num_hidden, recurrent=True, **params)

    learning_rate = RampUpDecaySchedule(embedding_size=dims, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_loss = tf.keras.metrics.Mean('train-loss', dtype=tf.float32)

    def compute_loss(sentences):
        y_hat, time_steps, targets, vectors = model(sentences, training=True, return_vectors=True)
        loss_val = sequence_cost(
            target_sequences=targets,
            sequence_logits=y_hat[:, :time_steps],
            num_labels=model.embed_layer.num_labels,
            smoothing=False
        )
        if orthogonal_cost:
            orthogonal_cost = vector_cost(context_vectors=vectors)
            return loss_val + orthogonal_cost
        return loss_val

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def train_step(sentences):
        loss_val = compute_loss(sentences)
        gradients = tf.gradients(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_val)  # log the loss value to TensorBoard

    model_file_name = None
    test_sentences = [
        "The movie was great!",
        "The movie was terrible."
    ]
    test_tokens = [' '.join(str_utils.clean_and_split(text)) for text in test_sentences]

    summary_writer_train = tf.summary.create_file_writer(log_dir + "/training")
    summary_writer_dev = tf.summary.create_file_writer(log_dir + "/validation")
    checkpoint = tf.train.Checkpoint(EmbeddingModel=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=5)

    # add word labels to the projector
    config = projector.ProjectorConfig()
    embeddings_config = config.embeddings.add()

    checkpoint_manager.save()
    reader = tf.train.load_checkpoint(log_dir)
    embeddings_config.tensor_name = [key for key in reader.get_variable_to_shape_map() if "embedding" in key][0]
    embeddings_config.metadata_path = log_dir + "/metadata.tsv"
    projector.visualize_embeddings(logdir=log_dir + "/training", config=config)

    step = 1
    for epoch in range(num_epochs):
        print(f"\t Epoch: {epoch + 1}")
        i = 1
        train_loss.reset_states()

        for x in mini_batches(train_corpus, size=batch_size):
            if len(x) == 0:
                continue

            if step == 1:
                tf.summary.trace_on(graph=True, profiler=False)

            train_step(x)
            with summary_writer_train.as_default():
                if step == 1:
                    tf.summary.trace_export(name='graph', step=1, profiler_outdir=log_dir)
                    tf.summary.trace_off()
                    summary_writer_train.flush()

                if i % log_step == 0:
                    print(f"\t\t iteration {i} - loss: {train_loss.result()}")
                    tf.summary.scalar(name='loss', data=train_loss.result(), step=step)
                    tf.summary.scalar(name='learning-rate', data=learning_rate.callback(step=step), step=step)
                    summary_writer_train.flush()
                    train_loss.reset_states()
            i += 1
            step += 1

        vectors = model.embed(test_tokens)
        angles = utils.compute_angles(vectors.numpy())
        for i, j in itertools.combinations(range(len(test_sentences)), r=2):
            print(f"The angle between '{test_sentences[i]}' and '{test_sentences[j]}' is {angles[i, j]} degrees")

        cv_loss = compute_loss(cv_tokens)
        with summary_writer_dev.as_default():
            tf.summary.scalar('loss', cv_loss.numpy(), step=step)
            tf.summary.scalar('similarity angle', angles[0, 1], step=step)
            summary_writer_dev.flush()
        model_file_name = checkpoint_manager.save()

    utils.log("Saving a frozen model")
    signatures = {"serving_default": model.embed, "token_embed": model.token_embed}
    tf.saved_model.save(model, f"{log_dir}/frozen/1", signatures)

    utils.log("Reloading frozen model and comparing output to in-memory model")
    test = tf.saved_model.load(f"{log_dir}/frozen/1")
    test_model = test.signatures["serving_default"]
    test_output = test_model(tf.constant(cv_tokens))["output_0"].numpy()
    utils.log(f"Outputs on CV set are approximately the same?: {np.allclose(test_output, model(cv_tokens).numpy())}")
    return model_file_name


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("run", choices=["train", "infer"], help="Run type.")
    parser.add_argument("model_name", type=str, help="Folder name in which to store model.")
    parser.add_argument("--tokens", type=int, help="Set the number of tokens to use.", default=10000)
    parser.add_argument("--embedding", type=int, help="Set the dimensionality of the word embeddings.", default=256)
    parser.add_argument("--hidden", type=int, help="Number of hidden model dimensions.", default=128)
    parser.add_argument("--layers", type=int, help="Number of self-attention layers.", default=8)
    parser.add_argument("--mb_size", type=int, help="Number of examples in each mini-batch.", default=32)
    parser.add_argument("--epochs", type=int, help="Number of epochs to run.", default=100000)
    parser.add_argument("--attention", action='store_true', help="Set to use attention transformer model.")
    parser.add_argument("--max_len", type=int, help="Maximum allowed sequence length", default=-1)
    parser.add_argument("--data_path", type=str, help="Path to the training data file(s).")
    parser.add_argument("--model_path", type=str, help="Path to place the saved model.")
    parser.add_argument("--orthogonal", action='store_true', help="Set to add a cost to mutually parallel contexts.")

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
            layers=args.layers,
            batch_size=args.mb_size,
            num_epochs=args.epochs,
            use_attention=bool(args.attention),
            data_path=args.data_path,
            model_path=args.model_path,
            orthogonal_cost=args.orthogonal
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
