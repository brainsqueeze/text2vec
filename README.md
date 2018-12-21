# text2vec

Models for contextual embedding of arbitrary texts.

## Motivation

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
before 
 

## Training

Both models are trained using Adam SGD with the learning-rate decay 
program in [[2](https://arxiv.org/abs/1706.03762)]. The default 
setting learns a word-embedding from scratch, however, it is possible 
to use a frozen GloVe embedding [[3](https://nlp.stanford.edu/pubs/glove.pdf)] 
instead.

Training the LSTM model can be initiated with
```bash
python -m bin.main train text_embedding --embedding 128 --hidden 128 --attention_size 64 --mb_size 32 --num_mb 40 --epochs 50
```
Likewise, the transformer model can be trained with 
```bash
python -m bin.main train text_embedding --embedding 128 --mb_size 32 --num_mb 40 --epochs 50 --attention
```

This command will read a text training set in from [data](text2vec/data) 
and takes the top 10,000 most frequent tokens. It them trains, and outputs 
the model into a folder that is dynamically created named `text_embedding`. 
To view the output of training you can then run
```bash
tensorboard --logdir text_embedding
```

If a pre-trained GloVe embedding is present, it can be used to seed 
the initial embedding matrix. To use the GloVe embeddings simply add these 
two flags to the training script:

```--use_glove --glove_file /absolute/path/to/glove/embeddings```

The default behavior for the text dictionary class is to take the 
top N tokens based strictly on term-frequencies in the corpus. It 
is possible to weight the tokens by TF-IDF values at training time 
and take the top N tokens based on the largest TF-IDF values. This 
can be done by passing the `--idf 1` flag to the training script.

If you have CUDA and cuDNN installed you can run 
`pip install -r requirements-gpu.txt`. 
The GPU will automatically be detected and used if present, otherwise 
it will fall back to the CPU for training and inferencing. 

To train a model with a custom data set, simply replace the 
data set(s) in [/data](text2vec/data) with your own.

#### Generalized cosine loss

A generalized cosine distance loss function, which is convex over 
the domain in this problem, is available for the LSMT-seq2seq model. 
The loss takes the functional form of

![equation](http://latex.codecogs.com/svg.latex?\mathcal{L}_{mb}%20=%20\sum_{j=1}^{N_{mb}}%20\sum_{i=1}^{L_j}%20\frac{1}{L_j}%20\left(1%20-%20\textbf{v}_i^{\mathcal{I}_j}%20\cdot%20\textbf{v}_i^{\mathcal{O}_j}%20\right%20))

where ![equation](http://latex.codecogs.com/svg.latex?\textbf{v}_i^{\mathcal{I}_j}) 
is the L2-normalized, d-dimensional input vector of the i-th sequence 
position of the j-th example in the mini-batch; 
![equation](http://latex.codecogs.com/svg.latex?\textbf{v}_i^{\mathcal{O}_j}) 
is the equivalent output from the decoding layers. 
![equation](http://latex.codecogs.com/svg.latex?L_j) is the sequence 
length of the j-th example. Since the model is an auto-morphism, then 
the loss function should approach 0 as training is allowed to continue.

It should be noted that this loss function will ultimately yield a worse 
model than the same model trained on the usual log-loss when word-embeddings 
are being learned. This is due to the loss function effectively teaching the
embedding matrix to map words to a single point. It has not been tested whether 
using pre-trained GloVe embeddings remedy this behavior.

## Inference Demo

Once a model is fully trained then a demo API can be run, along with a small 
UI to interact with the REST API. This demo attempts to use the trained model 
to condense long bodies of text into the most important sentences, using the 
inferred embedded context vectors.

To start the model server simply run 
```bash
python -m bin.main infer text_embedding
```
where `text_embedding` should be replaced with the name of your model's log 
directory.  A demonstration webpage is included in [demo](demo) at 
[context.html](demo/context.html).

## References

1. D. Bahdanau, K. Cho, Y. Bengio [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
2. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. J. Pennington, R. Socher, C. D. Manning [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
