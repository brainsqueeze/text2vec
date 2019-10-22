from text2vec.preprocessing import utils as str_utils
from text2vec.models import InputFeeder
from text2vec.models import TransformerEncoder, TransformerDecoder
from text2vec.optimizer_tools import RampUpDecaySchedule
from text2vec.training_tools import EncodingModel, train_step, get_context_embeddings, get_token_embeddings
from . import utils

import tensorflow as tf
import numpy as np

# from tensorboard.plugins import projector

import argparse
import os

root = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.compat.v1.enable_eager_execution()


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
          data_path=None, model_path=None, use_attention=False, verbose=False):
    """
    Core training algorithm
    :param model_folder: name of the folder to create for the trained model (str)
    :param num_tokens: number of vocab tokens to keep from the training corpus (int, optional)
    :param embedding_size: size of the word-embedding dimensions (int, optional)
    :param num_hidden: number of hidden LSTM dimensions (int, optional)
    :param max_allowed_seq: the maximum sequence length allowed, model will truncate if longer (int)
    :param attention_size: number of hidden attention-mechanism dimensions (int, optional)
    :param layers: number of multi-head attention mechanisms for transformer model (int, optional)
    :param batch_size: size of each mini-batch (int, optional)
    :param num_batches: number of mini-batches in each epoch (int, optional)
    :param num_epochs: number of training epochs (int, optional)
    :param data_path: valid path to the training data (str)
    :param model_path: valid path to where the model will be saved (str)
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
    hash_map, max_seq_len = str_utils.get_top_tokens(corpus, n_top=num_tokens)
    utils.log(f"Max sequence length: {max_seq_len}")

    with open(log_dir + "/metadata.tsv", "w") as tsv:
        for token, _ in sorted(hash_map.items(), key=lambda s: s[-1]):
            tsv.write(token + "\n")
        tsv.write("<unk>\n")

    keep_probabilities = [0.9, 0.75, 1.0]

    utils.log("Building computation graph")
    log_step = num_batches // 10
    size = len(hash_map)
    dims = embedding_size

    model = EncodingModel(
        feeder=InputFeeder(token_hash=hash_map, emb_dims=dims),
        encoder=TransformerEncoder(max_sequence_len=max_seq_len, embedding_size=dims),
        decoder=TransformerDecoder(max_sequence_len=max_seq_len, num_labels=size, embedding_size=dims)
    )
    # if use_attention:
    #     model = EncodingModel(
    #         feeder=InputFeeder(token_hash=hash_map, emb_dims=dims),
    #         encoder=TransformerEncoder(max_sequence_len=max_seq_len, embedding_size=dims),
    #         decoder=TransformerDecoder(max_sequence_len=max_seq_len, num_labels=len(hash_map), embedding_size=dims)
    #     )
    # else:
    #     raise NotImplementedError("Only the Transformer model is currently available")
    #     # model = models.Sequential(
    #     #     max_sequence_len=max_seq_len,
    #     #     embedding_size=embedding_size,
    #     #     token_hash=hash_map,
    #     #     num_hidden=num_hidden
    #     # )

    lstm_file_name = None
    # gpu_options = tf.compat.v1.GPUOptions(
    #     per_process_gpu_memory_fraction=0.8,
    #     allow_growth=True
    # )
    # sess_config = tf.compat.v1.ConfigProto(
    #     gpu_options=gpu_options,
    #     allow_soft_placement=True,
    #     log_device_placement=False
    # )

    if not tf.executing_eagerly():
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

    test_sentences = [
        "The movie was great!",
        "The movie was terrible."
    ]
    test_tokens = [' '.join(str_utils.clean_and_split(text)) for text in test_sentences]

    learning_rate = RampUpDecaySchedule(embedding_size=dims, warmup_steps=4000)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    summary_writer_train = tf.compat.v2.summary.create_file_writer(log_dir + "/training")
    summary_writer_dev = tf.compat.v2.summary.create_file_writer(log_dir + "/validation")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, log_dir, max_to_keep=5)

    # todo: get v2.0 API instructions
    # # add word labels to the projector
    # config_ = projector.ProjectorConfig()
    # embeddings_config = config_.embeddings.add()
    # embeddings = model.embed_layer.embeddings
    # embeddings_config.tensor_name = embeddings.name
    # embeddings_config.metadata_path = log_dir + "/metadata.tsv"
    # projector.visualize_embeddings(summary_writer_train, config_)

    train_loss = tf.keras.metrics.Mean('train-loss', dtype=tf.float32)
    # val_loss = tf.keras.metrics.Mean('validation-loss', dtype=tf.float32)

    step = 1
    for epoch in range(num_epochs):
        print(f"\t Epoch: {epoch + 1}")
        i = 1
        train_loss.reset_states()

        for x in mini_batches(train_corpus, size=batch_size, n_batches=num_batches, seed=epoch):
            if len(x) == 0:
                continue

            if i == log_step:
                tf.compat.v2.summary.trace_on(graph=True, profiler=True)
            loss, _ = train_step(x, model=model, optimizer=optimizer)
            train_loss(loss)  # log the loss value to TensorBoard

            if i % log_step == 0:
                print(f"\t\t iteration {i} - loss: {train_loss.result()}")
                with summary_writer_train.as_default():
                    tf.compat.v2.summary.scalar('train-loss', train_loss.result(), step=step)
                    tf.compat.v2.summary.scalar('learning-rate', learning_rate(step=step), step=step)

                    if i == log_step:
                        tf.compat.v2.summary.trace_export('train-step-trace', step=step, profiler_outdir=log_dir)

        vectors = model.encode_layer(test_tokens, training=False)
        angle = utils.compute_angles(vectors.numpy())[0, 1]
        print(f"The 'angle' between `{'` and `'.join(test_sentences)}` is {angle} degrees")

        cv_loss = model(cv_tokens)
        with summary_writer_dev.as_default():
            tf.compat.v2.summary.scalar('validation-loss', cv_loss.numpy(), step=step)
            tf.compat.v2.summary.scalar('similarity angle', angle, step=step)
        lstm_file_name = checkpoint_manager.save()

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
    parser.add_argument("--attention", action='store_true', help="Set to use attention transformer model.")
    parser.add_argument("--max_len", type=int, help="Maximum allowed sequence length", default=-1)
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
