# text2vec

Contextual embedding for continuous texts.

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
models (pointer-generator), where an attention mechanism is 
used to extrude the overall meaning of the input text. In the 
case of text2vec, we use the attention vectors found from the 
input text as the embedded vector representing the input.

**note**: this is not a canonical implementation of the attention 
mechanism, but this method was chosen intentionally to be able to 
leverage the attention vector as the embedding output.

## Training

Executing the training algorithm is accomplished by 
```bash
python -m bin.main train \
                   text_embedding \
                   --hidden 128 \
                   --attention_size 64 \
                   --mb_size 32 \
                   --num_mb 40 \
                   --epochs 50
```
This command will read a text training set in from [data](text2vec/data) 
and takes the top 10,000 most frequent tokens. It compiles 
a LSTM+attention encoder-decoder automorphism model with 
128 hidden LSTM layers in both the encoder/decoder layers, 
and 128 dimensions for the weights of the attention 
mechanism. It them trains the model over 10,000 epochs, and 
outputs the model into a folder that is dynamically created 
named `text_embedding`. To view the output of training you 
can then run
```bash
tensorboard --logdir text_embedding
```

To train a model with a custom data set, simply replace the 
data set(s) in [/data](text2vec/data) with your own.

The model learns by minimizing on a generalized cosine distance 
loss function, which is convex over the domain in this problem. 
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
