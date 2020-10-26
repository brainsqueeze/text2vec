# text2vec

Models for contextual embedding of arbitrary texts.

## Setup
---

For the GPU build of Tensorflow, if nightly features are not
required it is recommended to install Tensorflow and its dependencies
through Anaconda as
```bash
conda install -c anaconda tensorflow-gpu
```

To install the core components as an import-able Python library
simply run

```bash
pip install 'text2vec[device] @ git+https://github.com/brainsqueeze/text2vec.git'
```
where `device` is either `cpu` or `gpu`, depending on which flavor
of TensorFlow one wishes to install.

## Motivation
---

Word embedding models have been very beneficial to natural 
language processing. The technique is able to distill semantic 
meaning from words by treating them as vectors in a 
high-dimensional vector space.

This package attempts to accomplish the same semantic embedding, 
but do this at the sentence and paragraph level. Within a 
sentence the order of words and the use of punctuation and 
conjugations are very important for extracting the meaning 
of blocks of text.

Inspiration is taken from recent advances in text summary 
models (pointer-generator), where an attention mechanism 
[[1](https://arxiv.org/abs/1409.0473)] is 
used to extrude the overall meaning of the input text. In the 
case of text2vec, we use the attention vectors found from the 
input text as the embedded vector representing the input. 
Furthermore, recent attention-only approaches to sequence-to-sequence 
modeling are adapted.

**note**: this is not a canonical implementation of the attention 
mechanism, but this method was chosen intentionally to be able to 
leverage the attention vector as the embedding output.

### Transformer model
---

This is a tensor-to-tensor model adapted from the work in 
[Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
The embedding and encoding steps follow directly from 
[[2](https://arxiv.org/abs/1706.03762)], however a self-
attention is applied at the end of the encoding steps and a 
context-vector is learned, which in turn is used to project 
the decoding tensors onto.

The decoding steps begin as usual with the word-embedded input 
sequences shifted right, then multi-head attention, skip connection 
and layer-normalization is applied. Before continuing, we project 
the resulting decoded sequences onto the context-vector from the 
encoding steps. The projected tensors are then passed through 
the position-wise feed-forward (conv1D) + skip connection and layer- 
normalization again, before once more being projected onto the 
context-vectors.

### LSTM seq2seq

This is an adapted bi-directional LSTM encoder-decoder model with 
a self-attention mechanism learned from the encoding steps. The 
context-vectors are used to project the resulting decoded sequences 
before computing logits.
 

## Training
---

Both models are trained using Adam SGD with the learning-rate decay 
program in [[2](https://arxiv.org/abs/1706.03762)].

Training the LSTM model can be initiated with
```bash
text2vec_main --run=train --yaml_config=/path/to/config.yml
```
The training configuration YAML for attention models must look like
```yaml
training:
  tokens: 10000
  max_sequence_length: 512
  epochs: 100
  batch_size: 64

model:
  name: transformer_test
  parameters:
    embedding: 128
    layers: 8
  storage_dir: /path/to/save/model
```
The `parameters` for recurrent models must include at least 
`embedding` and `hidden`, which referes to the dimensionality of the hidden LSTM layer. The `training` section of the YAML file can also include user-defined sentences to use as a context-angle evaluation set. This can look like
```yaml
eval_sentences:
  - The movie was great!
  - The movie was terrible.
```
It can also include a `data` tag which is a list of absolute file paths for custom training data sets. This can look like
```yaml
data_files:
  - ~/path/to/data/set1.txt
  - ~/path/to/data/set2.txt
  ...
```

Likewise, the transformer model can be trained with 
```bash
text2vec_main --run=train --attention --yaml_config=/path/to/config.yml
```

To view the output of training you can then run
```bash
tensorboard --logdir text_embedding
```

If you have CUDA and cuDNN installed you can run 
`pip install -r requirements-gpu.txt`. 
The GPU will automatically be detected and used if present, otherwise 
it will fall back to the CPU for training and inferencing.

### Mutual contextual orthogonality

To impose quasi-mutual orthogonality on the learned context vectors simply add the `--orthogonal` flag to the training command. This will add a loss term that can be thought of as a Lagrange multiplier where the constraint is self-alignment of the context vectors, and orthogonality between non-self vectors. The aim is not to impose orthogonality between all text inputs that are not the same, but rather to coerce the model to learn significantly different encodings for different contextual inputs.

## Python API

Text2vec includes a Python API with convenient classes for handling attention and LSTM mechanisms. These classes can be used to create custom models and layers to address custom problems.

### Model components

#### Pre-built Models

  - [text2vec.models.TransformerEncoder](/text2vec/models/transformer.py#11)
  - [text2vec.models.TransformerDecoder](/text2vec/models/transformer.py#81)
  - [text2vec.models.RecurrentEncoder](/text2vec/models/sequential.py#8)
  - [text2vec.models.RecurrentDecoder](/text2vec/models/sequential.py#61)

#### Input and Word-Embeddings Components

  - [text2vec.models.TextInput](/text2vec/models/components/feeder.py#L35)
  - [text2vec.models.Tokenizer](/text2vec/models/components/feeder.py#L4)

#### Attention Components

  - [text2vec.models.components.attention.ScalarDotAttention](/text2vec/models/components/attention.py#L4)
  - [text2vec.models.components.attention.SingleHeadAttention](/text2vec/models/components/attention.py#L111)
  - [text2vec.models.MultiHeadAttention](/text2vec/models/components/attention.py#L175)
  - [text2vec.models.BahdanauAttention](/text2vec/models/components/attention.py#L53)

#### LSTM Components

  - [text2vec.models.BidirectionalLSTM](/text2vec/models/components/recurrent.py#L4)

#### Pointwise Feedforward Components

  - [text2vec.models.PositionWiseFFN](/text2vec/models/components/feed_forward.py#L4)

#### General Layer Components

  - [text2vec.models.components.utils.LayerNorm](/text2vec/models/components/utils.py#5)
  - [text2vec.models.components.utils.TensorProjection](/text2vec/models/components/utils.py#43)
  - [text2vec.models.components.utils.PositionalEncder](/text2vec/models/components/utils.py#76)

#### Dataset Pre-processing
  
  - [text2vec.preprocessing.get_top_tokens](/text2vec/preprocessing/utils.py#5)

#### String Pre-processing

  - [text2vec.preprocessing.text.clean_and_split](/text2vec/preprocessing/text.py#6)
  - [text2vec.preprocessing.text.replace_money_token](/text2vec/preprocessing/text.py#27)
  - [text2vec.preprocessing.text.replace_urls_token](/text2vec/preprocessing/text.py#43)
  - [text2vec.preprocessing.text.fix_unicode_quotes](/text2vec/preprocessing/text.py#60)
  - [text2vec.preprocessing.text.format_large_numbers](/text2vec/preprocessing/text.py#78)
  - [text2vec.preprocessing.text.pad_punctuation](/text2vec/preprocessing/text.py#95)
  - [text2vec.preprocessing.text.normalize_text](/text2vec/preprocessing/text.py#113)


## Inference Demo
---

Once a model is fully trained then a demo API can be run, along with a small 
UI to interact with the REST API. This demo attempts to use the trained model 
to condense long bodies of text into the most important sentences, using the 
inferred embedded context vectors.

To start the model server simply run 
```bash
text2vec_main --run=infer --yaml_config=/path/to/config.yml
```
A demonstration webpage is included in [demo](demo) at 
[context.html](demo/context.html).

## References
---

1. D. Bahdanau, K. Cho, Y. Bengio [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
2. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
