<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$','$'], ['\\(','\\)']],
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
        }
    });
    MathJax.Hub.Queue(function() {
        var all = MathJax.Hub.getAllJax(), i;
        for(i = 0; i < all.length; i += 1) {
            all[i].SourceElement().parentNode.className += ' has-jax';
        }
    });
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

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
leverage the attention vector as the embedding output

## Training

Executing the training algorithm is accomplished by 
```bash
python -m bin.main run=train \
       --tokens 10000 \
       --hidden 128 \
       --attention_size 128 \
       --epochs 10000 \
       --model_name text_embedding
```
This command will read a text training set in from [/data](text2vec/data) 
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

$\mathcal{L}_{mb} = \sum_{j=0}^{N_{mb}} \frac{\sum_{i=0}^{N_{d}} 1 - v_i^{j^\mathcal{I}} \cdot v_i^{j^\mathcal{O}} }{L_j}$
