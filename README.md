# text2vec

Models for contextual embedding of arbitrary texts.

## Setup
---

To get started, one should have a flavor of TensorFlow installed, with version `>=2.4.1`. One can run
```bash
pip install tensorflow>=2.4.1
```
If one wishes to run the examples, some additional dependencies from HuggingFace will need to be installed. The full installation looks like
```bash
pip install tensorflow>=2.4.1 tokenizers datasets
```

To install the core components as an import-able Python library simply run

```bash
pip install git+https://github.com/brainsqueeze/text2vec.git
```

## Motivation
---

Word embedding models have been very beneficial to natural language processing. The technique is able to distill semantic meaning from words by treating them as vectors in a high-dimensional vector space.

This package attempts to accomplish the same semantic embedding, but do this at the sentence and paragraph level. Within a sentence the order of words and the use of punctuation and conjugations are very important for extracting the meaning of blocks of text.

Inspiration is taken from recent advances in text summary models (pointer-generator), where an attention mechanism [[1](https://arxiv.org/abs/1409.0473)] is used to extrude the overall meaning of the input text. In the case of text2vec, we use the attention vectors found from the input text as the embedded vector representing the input. Furthermore, recent attention-only approaches to sequence-to-sequence modeling are adapted.


### Transformer model
---

This is a tensor-to-tensor model adapted from the work in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The embedding and encoding steps follow directly from [[2](https://arxiv.org/abs/1706.03762)], however a self-attention is applied at the end of the encoding steps and a context-vector is learned, which in turn is used to project the decoding tensors onto.

The decoding steps begin as usual with the word-embedded input sequences shifted right, then multi-head attention, skip connection and layer-normalization is applied. Before continuing, we project the resulting decoded sequences onto the context-vector from the encoding steps. The projected tensors are then passed through the position-wise feed-forward (conv1D) + skip connection and layer-normalization again, before once more being projected onto the context-vectors.

### LSTM seq2seq

This is an adapted bi-directional LSTM encoder-decoder model with a self-attention mechanism learned from the encoding steps. The context-vectors are used to project the resulting decoded sequences before computing logits.
 

## Training
---

Both models are trained using Adam SGD with the learning-rate decay program in [[2](https://arxiv.org/abs/1706.03762)].

The pre-built auto-encoder models inherit from [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model), and as such they can be trained using the [fit method](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit). Training examples are available in the [examples folder](./examples/trainers). This uses HuggingFace [tokenizers](https://huggingface.co/docs/tokenizers/python/latest/) and [datasets](https://huggingface.co/docs/datasets/master/).

If you wish to run the example training scripts then you will need to clone the repository
```bash
git clone https://github.com/brainsqueeze/text2vec.git
```
and then run either 
```bash
python -m examples.trainers.news_transformer
```
for the attention-based transformer, or 
```bash
python -m examples.trainers.news_lstm
```
for the LSTM-based encoder. These examples use the [Multi-News](https://github.com/Alex-Fabbri/Multi-News) dataset via [HuggingFace](https://huggingface.co/datasets/multi_news).


## Python API

Text2vec includes a Python API with convenient classes for handling attention and LSTM mechanisms. These classes can be used to create custom models and layers to address custom problems.

### Model components

#### Auto-encoders

  - [text2vec.autoencoders.TransformerAutoEncoder](/text2vec/autoencoders.py#L12)
  - [text2vec.autoencoders.LstmAutoEncoder](/text2vec/models/transformer.py#L190)

#### Layers

  - [text2vec.models.TransformerEncoder](/text2vec/models/transformer.py#L8)
  - [text2vec.models.TransformerDecoder](/text2vec/models/transformer.py#L78)
  - [text2vec.models.RecurrentEncoder](/text2vec/models/sequential.py#L9)
  - [text2vec.models.RecurrentDecoder](/text2vec/models/sequential.py#L65)

#### Input and Word-Embeddings Components

  - [text2vec.models.Tokenizer](/text2vec/models/components/text_inputs.py#L5)
  - [text2vec.models.Embed](/text2vec/models/components/text_inputs.py#L36)
  - [text2vec.models.TokenEmbed](/text2vec/models/components/text_inputs.py#L116)

#### Attention Components

  - [text2vec.models.components.attention.ScaledDotAttention](/text2vec/models/components/attention.py#L7)
  - [text2vec.models.components.attention.SingleHeadAttention](/text2vec/models/components/attention.py#L115)
  - [text2vec.models.MultiHeadAttention](/text2vec/models/components/attention.py#L179)
  - [text2vec.models.BahdanauAttention](/text2vec/models/components/attention.py#L57)

#### LSTM Components

  - [text2vec.models.BidirectionalLSTM](/text2vec/models/components/recurrent.py#L5)

#### Pointwise Feedforward Components

  - [text2vec.models.PositionWiseFFN](/text2vec/models/components/feed_forward.py#L4)

#### General Layer Components

  - [text2vec.models.components.utils.LayerNorm](/text2vec/models/components/utils.py#L6)
  - [text2vec.models.components.utils.TensorProjection](/text2vec/models/components/utils.py#L43)
  - [text2vec.models.components.utils.PositionalEncder](/text2vec/models/components/utils.py#L77)
  - [text2vec.models.components.utils.VariationPositionalEncoder](/text2vec/models/components/utils.py#L122)

#### Dataset Pre-processing
  
  - [text2vec.preprocessing.utils.get_top_tokens](/text2vec/preprocessing/utils.py#L9)
  - [text2vec.preprocessing.utils.check_valid](/text2vec/preprocessing/utils.py#L46)
  - [text2vec.preprocessing.utils.load_text_files](/text2vec/preprocessing/utils.py#L68)

#### String Pre-processing

  - [text2vec.preprocessing.text.clean_and_split](/text2vec/preprocessing/text.py#L6)
  - [text2vec.preprocessing.text.replace_money_token](/text2vec/preprocessing/text.py#L27)
  - [text2vec.preprocessing.text.replace_urls_token](/text2vec/preprocessing/text.py#L43)
  - [text2vec.preprocessing.text.fix_unicode_quotes](/text2vec/preprocessing/text.py#L60)
  - [text2vec.preprocessing.text.format_large_numbers](/text2vec/preprocessing/text.py#L78)
  - [text2vec.preprocessing.text.pad_punctuation](/text2vec/preprocessing/text.py#L95)
  - [text2vec.preprocessing.text.normalize_text](/text2vec/preprocessing/text.py#L113)


## Inference Demo
---

Trained text2vec models can be demonstrated from a lightweight app included in this repository. The demo runs extractive summarization from long bodies of text using the attention vectors of the encoding latent space. To get started, you will need to clone the repository and then install additional dependencies:
```bash
git clone https://github.com/brainsqueeze/text2vec.git
cd text2vec
pip install flask tornado
```
To start the model server simply run 
```bash
python demo/api.py --model_dir /absolute/saved_model/parent/dir
```
The `model_dir` CLI parameter must be an absolute path to the directory containing the `/saved_model` folder and the `tokenizer.json` file from a text2vec model with an `embed` signature. A demonstration app is served on port 9090.

## References
---

1. D. Bahdanau, K. Cho, Y. Bengio [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
2. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
