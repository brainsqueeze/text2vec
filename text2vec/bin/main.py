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
from glob import glob
import itertools
import argparse
import yaml
import os

root = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def check_valid(text, max_length):
    sequence_lengths = tf.shape(tf.strings.split(text, sep=''))
    if max_length < -1:
        return sequence_lengths is not None
    return sequence_lengths is not None and sequence_lengths[0] <= max_length


def load_text(data_files=None, max_length=-1):
    """
    Loads the training data from a text file
    :param data_files: list of absolute paths to training data set files (list)
    :param max_length: maximum sequence length to allow (int)
    :return: (list)
    """

    files = []
    if isinstance(data_files, list) and len(data_files) > 0:
        for f in data_files:
            if '*' in f:
                files.extend(glob(f))
                continue
            if os.path.isfile(f):
                files.append(f)
    else:
        path = f"{root}/../../text2vec/data/"
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    texts = tf.data.TextLineDataset(files)
    texts = texts.map(lambda x: tf.strings.strip(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return texts.filter(lambda x: check_valid(x, max_length))


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
          layers=8, batch_size=32, num_epochs=10, data_files=None, model_path=".", use_attention=False,
          eval_sentences=None, orthogonal_cost=False):
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
    :param data_files: list of absolute paths to training data sets (list)
    :param model_path: valid path to where the model will be saved (str)
    :param use_attention: set to True to use the self-attention only model (bool)
    :param eval_sentences: list of sentences to check the context angles (list)
    :param orthogonal: set to True to add a cost to mutually parallel context vector (bool)
    """

    # GPU config
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_soft_device_placement(True)

    log_dir = f"{model_path}/{model_folder}" if model_path else f"{root}/../../text2vec/{model_folder}"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    utils.log("Fetching corpus and transforming to frequency domain")
    corpus = load_text(data_files=data_files, max_length=max_allowed_seq)
    assert isinstance(corpus, tf.data.Dataset)

    utils.log("Fitting embedding lookup and transforming the training and cross-validation sets")
    hash_map, max_seq_len, train_set_size = str_utils.get_top_tokens(corpus, n_top=num_tokens)
    utils.log(f"Max sequence length: {max_seq_len}")

    with open(log_dir + "/metadata.tsv", "w") as tsv:
        for token, _ in sorted(hash_map.items(), key=lambda s: s[-1]):
            tsv.write(token + "\n")
        tsv.write("<unk>\n")

    utils.log("Building computation graph")
    log_step = (train_set_size // batch_size) // 10
    dims = embedding_size

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

    warmup_steps = max(train_set_size // batch_size, 4000)
    learning_rate = RampUpDecaySchedule(embedding_size=dims, warmup_steps=warmup_steps)
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
            return loss_val + vector_cost(context_vectors=vectors)
        return loss_val

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.string)])
    def train_step(sentences):
        loss_val = compute_loss(sentences)
        gradients = tf.gradients(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss_val)  # log the loss value to TensorBoard

    model_file_name = None
    if isinstance(eval_sentences, list) and len(eval_sentences) > 1:
        test_sentences = eval_sentences
    else:
        test_sentences = ["The movie was great!", "The movie was terrible."]
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
        try:
            corpus = corpus.unbatch()
        except ValueError:
            print("Corpus not batched")
        corpus = corpus.shuffle(train_set_size)
        corpus = corpus.batch(batch_size).prefetch(10)  # pre-fetch 10 batches for queuing

        print(f"\t Epoch: {epoch + 1}")
        i = 1
        train_loss.reset_states()

        for x in corpus:
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

        with summary_writer_dev.as_default():
            for idx, (i, j) in enumerate(itertools.combinations(range(len(test_sentences)), r=2), start=1):
                angle = angles[i, j]
                print(f"The angle between '{test_sentences[i]}' and '{test_sentences[j]}' is {angle} degrees")

                # log the angle to tensorboard
                desc = f"'{test_sentences[i]}' : '{test_sentences[j]}'"
                tf.summary.scalar(f'similarity-angle/{idx}', angle, step=step, description=desc)
            summary_writer_dev.flush()
        model_file_name = checkpoint_manager.save()

    utils.log("Saving a frozen model")
    signatures = {"serving_default": model.embed, "token_embed": model.token_embed}
    tf.saved_model.save(model, f"{log_dir}/frozen/1", signatures)

    utils.log("Reloading frozen model and comparing output to in-memory model")
    test = tf.saved_model.load(f"{log_dir}/frozen/1")
    test_model = test.signatures["serving_default"]
    test_output = test_model(tf.constant(test_tokens))["output_0"].numpy()
    utils.log(f"Outputs on CV set are approximately the same?: {np.allclose(test_output, model(test_tokens).numpy())}")
    return model_file_name


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", choices=["train", "infer"], help="Run type.", required=True)
    parser.add_argument("--attention", action='store_true', help="Set to use attention transformer model.")
    parser.add_argument("--orthogonal", action='store_true', help="Set to add a cost to mutually parallel contexts.")
    parser.add_argument("--yaml_config", type=str, help="Path to a valid training config YAML file.", required=True)
    args = parser.parse_args()

    config_path = args.yaml_config
    if config_path.startswith("${HOME}"):
        config_path = config_path.replace('${HOME}', os.getenv('HOME'))
    elif config_path.startswith("$HOME"):
        config_path = config_path.replace('$HOME', os.getenv('HOME'))

    config = yaml.safe_load(open(config_path, 'r'))
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    model_params = model_config.get("parameters", {})

    if args.run == "train":
        train(
            model_folder=model_config["name"],
            use_attention=args.attention,
            num_tokens=training_config.get("tokens", 10000),
            max_allowed_seq=training_config.get("max_sequence_length", 512),
            embedding_size=model_params.get("embedding", 128),
            num_hidden=model_params.get("hidden", 128),
            layers=model_params.get("layers", 8),
            batch_size=training_config.get("batch_size", 32),
            num_epochs=training_config.get("epochs", 20),
            data_files=training_config.get("data_files"),
            model_path=model_config.get("storage_dir", "."),
            eval_sentences=training_config.get("eval_sentences"),
            orthogonal_cost=args.orthogonal
        )
    elif args.run == "infer":
        os.environ["MODEL_PATH"] = f'{model_config.get("storage_dir", ".")}/{model_config["name"]}'
        from .text_summarize import run_server
        run_server(port=8008)
    else:
        raise NotImplementedError("Only training and inferencing is enabled right now.")
    return


if __name__ == '__main__':
    main()
