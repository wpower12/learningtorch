# Learning Torch
These are notes from working a few tutorials on the torch framework. Each one has a few code examples, and usually one exercise. The code examples were mostly hand copied, with my own comments, and a few tiny style things/constant definitions added. The exercise code files end in `ex` and start from w.e starting material the tutorial provides. I tried to stick to torch only/appropriate use of the available modules. 

## Torch Website - Simple Tutorials

The tutorial on the actual torch site does an interesting thing and walks you through building the same small network in a few different ways. First, it shows you the network as if you only used numpy to build the model. Then, using the basics of torch. In the second version, you hand craft both the forward and backwards passes. The third version uses torch as well, but the backwards pass is handled by the automatic gradient framework built into torch. 

Then it goes into building your own autograd-equipped method to operate on tensors. This is all very straightforward. This is a great set of tutorials.

For future-Bill: the tutorials are [here](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) (as of May 2019). Each of the following subsections are one of the 4 tutorials on that site. 

Files related to this tutorial:

* `tut_numpy.py`
* `tut_torch1_noauto.py`
* `tut_torch2_autograd.py`
* `tut_torch3_custauto.py`

### Only numpy 

Straight forward! The backwards pass is us manually performing the symoblic differentiation one would do if you were computing the gradients 'by hand'. Of note are a few things I didn't know I could do in numpy, specifically the indexing done during the relu backprop - the derivative of the ReLU unit is either 0, or the value itself, so to do this, the code indexes a copy of the relu results with an implicit array of booleans based on the values of h. (line 39 in the code of `tut_numpy.py`).

### Basic torch

This big difference is the commands that tell torch what type of device, and what type of data is being manipulated. It looks like this means torch lets you easily and quickly make use of the GPU based optimizations available with things like CUDA. The code itself is almost identical to the numpy example. I guess thats the point! 

The syntax for matrix operations is a little different. Instead of `tensor.dot(..)` you use `tensor.mm(..)` for the typical matrix multiplication. Transposes are done with `tensor.t()` instead of the simple `tensor.T`. I think this is due to the static vs dynamic thing. Since torch uses dynamic graphs, the transpose is a method, versus numpy, where the static graph can 'store' a copy of its transpose? Doesn't much matter. 

### Autograd torch

Time for automatic differentiation! The big changes come from indicating in the constructors for the tensors which one will have gradients computed for them. This is called out in the code.

The confusing part is why we need to do the `no_grad()` stuff. I think its a context thing, but I'm not sure. Note, we are only getting the result of the backwards pass for free, its still on us to do all the things with this, like update weights. Makes sense. 

Stupid easy to use. Wow.

### Custom autograd torch

To define out own functions that can have their own gradients computed during a backwards pass, we can extend a torch provided class. Pretty easy compared to the tensorflow framework. 

## Torch Website - Word Embeddings

Tutorial found [here](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html).

Files related to this tutorial:

* `tut_torch4_ngram.py`
* `tut_torch5_cbow_ex.py`

More in depth tutorial, that also touches on the actual theory of these in a light, digestable way. First off, what is the problem that word embeddings are solving? Quite simply, when we want to represent a word from a dictionary or corpus within a language, how do we get a representation that captures a notion of "similarity" between two words? The basic 'bag of words' ignores all notion of similarity. The location of a word in the space has no bearing on its semantic relation to the other words in the bag. 

Embeddings give us this notion. Example sentences are used to build an understanding of the joint distribution of words that are present together in a sentence, based on their relative locations within that sentence (its context). The ability to do this rests on the assumption that the semantics of a language indeed follow a joint distribution in this manner. The tutorial links to the wikipedia article on the "[Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics)" The general idea is summed up in the quote "linguistic items with similar distributions have similar meanings."

How do we do this? If we could, we'd enumerate a list of possible features, then, using the distributions of words in sentences, fit each word to a set of values for these features. But we can abuse deep learning! Why not let the network learn what the good features are, then embed in that space? Thats what we do. The learned parameters are those that project a word, based on its joint distribution with other words, into a similarity preserving space. 

We want to find these latent semantic features with a network. 

### Example - NGram Approach
Want to embed by assuming we can predict a word based on some preceeding words. To train, we give the network a concatenation of the one-hot vectors of the input n-grams, project them into a low dimensional space (recall: project == multiply by a matrix, which are our parameters), then project from that space back to one with the size of the vocab. So the model is trying to predict the last word, given all the others. 

The code example shows you how to get the trigrams from an example, and the bookkeeping needed to track the words, and their indexes in a dictionary. This is needed because the model only cares about indexes, not actual words. We recover those at the end with the map. 

In the `init` method for the module, we see that the archtiecture of the model spelled out. The `linear1` and `linear2` tensors are the weights that handle the projection from the input space to the latent space, and then from the latent space to the space of indexes. The input space is of the size `context_size * embedding_dim`, which makes intuitive sense, we have a concatenated vector. You see this in the `forward` method where the input indexes are passed to the embedding, which yeilds a tensor which is reshaped to build the concatenated embedding in the latent space. 

Ok its a day later and I want to amend the above writing. The linear1 and linear2 are NOT tensors, they are extensions of `nn.Module` in their own right. It looks like being a module means you implicitly expose your `grad_required` fields/ops to the `.parameters()` method. This allows the other things like autograd and optimizers to work on the actual parameters of your model. Very cool, very interesting method of handling modularity and extensibility. 

### Exercise - Context Bag of Words Approach

The model is that we look at the context of the word, and sum up embedded vectors. These are then used to predict the target word using a single linear function: Ax+b.

I got it working! It looks like the losses are actually going down, as intended. There were a couple hang ups while getting the tensors to be the right shape. Mostly it was with respect to getting the right shape of the summed embedding vectors. This was easy once I added some print statements that told me the shapes lol. The trick was to get the transpose of the sumed vectors with the `view((1, -1))` method. I just followed along from the other example to get this right. 

Also of note, no need to muck about with my own tensors and telling the system I want them to have gradients calculated. The `linear` module does exactly what we want: `x*A.T + b`. So I just used that instead of trying to get anything working by hand.  

## Torch Website - Sequence Models and LSTM Nets

The tutorial can be found [here](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py).

Files related to this tutorial:

* `tut_torch6_simpleLSTM.py`
* `tut_torch7_posttagging.py`
* `tut_torch8_pos_wchar_ex.py`

### Simple LSTM Example
This mostly goes over the specifics of how torch does sequence data. Of note is the fact that the sequence modules (like the `nn.LSTM` one) can handle a sequence all at once. It's still doing the normal sequence/RNN thing where the items are passed sequentially, and the output and hidden on to the next item's cell. But, now you can just pass a tensor of the entire sequence. The output of these will give you the entire histroy of hiddens and outputs, or just the last one, however you'd like to use it. You can still manually do the sequenceing yourself, or let torch handle it for you. 

The format of this is a 3D tensor, of the following 'shape': |seq|x|mini-batch|x|seq element|. For this example, and most "simple" recurrent/sequence models, the size of a sequence element will be 1 (like 1 word, or 1 character). For us, the size of the mini_batch will be 1 as well. Not sure how you handle mini batches of sequences of different length but its there? The tut doesn't go into it. Should maybe look into the way torch deals with mini-batching.

