---
layout: post
title: "Learning to Implement Papers: Neural Language Model in Tensorflow from Scratch"
date: 2017-06-05
comments: true
---

# Intro

Learning how to implement a paper is a skill often ignored by people first starting out in NLP, but it is this skill in which one of the core tenants of science takes place: reproducibility. Learning to implement a paper will not only help you as a programer and a researcher, but it will help the science progress as well. Implementing a paper can be intimidating (dense academic text doesn't exactly translate easily to code) but sitting down with a paper for a while and really going through the process of parsing exactly what needs to happen where is the best way to learn about a topic. For this post I will be implementing the seminal paper from [Yoshua Bengio et al., 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), which introduced the possibility of using a Neural Network to create a highly accurate Language Model. The use of Neural Networks for Language Modeling (which will be explained in the next section, don't worry) is now the state of the art in Language Modeling.

## Language Models

What exactly is the inscrutably named Language Model? A [Language Model](https://en.wikipedia.org/wiki/Language_model) is a probability distribution over sequences of words. If you don't have a statistics/probability background however, this won't mean anything at all, so let's focus on the intuition a bit before we jump into the math of it. Say you were trying to translate a sentence from French into English. The French sentence could be something like: "J'aime bien la PNL!". The problem with translating this is that there are a huge number of ways that one _could_ translate it, for example: "I love NLP!" or "I really like NLP!" or even "I like well the NLP!". What we need to be able to do is take all these possible translations and ask which one sounds the most "English". That's where the Language Model comes in. The ideal Language Model would be able to tell you that "I love NLP!" is way more "English" than "I like well the NLP!", despite the latter having a closer correspondance with the actual French words.

Great, so now that we know what a Language Model _does_ we need to ask ourselves _how_ a Language Model does it. The old tried and true methods are called _N_-gram Language Models, and they actually are pretty much always a competitive baseline. Essentially the way that they work is they take an _n_-gram (an ordered tuple of _n_ words) and they estimate the probability of a sequence of words as the product of the probabilities of all of it's constituent _n_-grams. More intuitively, if you are given the sentence: "The cat is pretty neat", you can break it down into _bigrams_ (_n_-grams with 2 items) which would look like this: (The, cat), (cat, is), (is, pretty), (pretty, neat). From here, the probability of the entire string is just the product of the probability of each bigram, which is calculated from some larger corpus of data as: 

` # of times W2 follows W1 / total # of times W1 is used `

or intuitively, the relative frequency that W2 is used after W1. In practice, these _n_-gram Language Models has a significant drawback: the larger the corpus the amount of possible _n_-grams grows exponentially. In short, we need a way of calculating the probability of a sentence which alleviates this [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

## Towards a Neural Solution

The solution to this problem as proposed in [Bengio et al., 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) is to use distributed representations of words and train those representations and a joint probability function via a Neural Network. Unfortunately an in depth explaination of what a Neural Network is is beyond the scope of this blog post, so I will simply direct you to some great resources [here](http://neuralnetworksanddeeplearning.com/). An incredibly simple explaination would be that for each word w<sub>i</sub>, a _d_-dimensional vector v<sub>i</sub> is used to represent it. Initially these vectors are randomly assigned, but during the training of the Network they are adjusted to minimize some loss function J. The Network attempts to compute the probability of a word w<sub>i+k+1</sub> appearing as the word after a _k_ length window of words w<sub>i</sub>, w<sub>i+1</sub>, ..., w<sub>i+k</sub>. By training to maximize the log-likelihood of the correct w<sub>i+k+1</sub>, the Network adjusts the parameters of the prediction function and the parameters of the vector representations to reflect the distributional statistics of the corpus. The end result is a Network that is capable of computing the probability of w<sub>i+k+1</sub> coming after a _k_-gram (look familiar?) and continuous fixed-dimensional representations of words. If none of the above made sense to you that's fine, it will be explored more in depth in the next section where we begin to implement this model in [Tensorflow](https://www.tensorflow.org/).

# Implementation

We will break the implementation itself up into the constituent parts that the paper outlines:

1. Associate each word in the vocabulary with a distributed _word feature vector_ (a real-valued vector in R<sup>m</sup>), 
2. Express the joint _probability function_ of word sequences in terms of the feature vectors of these words in the sequence, and 
3. Learn simultaneously the _word feature vectors_ and the parameters of that _probability function_. 

-- (Bengio et al., 2003, pg. 3)

Here is a diagram of the whole network in place:

![](/assets/bengio_fig_1.png)

## Part 1: Word Feature Vectors

For this section we need to be able to create a trainable `|V| x d` matrix, where `|V|` is the size of our vocabulary and `d` is the dimensionality we want our word feature vectors to be. Luckily, in tensorflow this is super easy.

```
import tensorflow as tf
import numpy as np

embed_matrix = tf.Variable(tf.random_uniform([len(V), embed_dim], -1.0, 1.0),name='embed_matrix')
```

And it's that easy! The variable in tensorflow is a trainable tensor that will be updated in our optimization step later on. We initialized it with a sample from the random uniform distribution between -1 and 1 using the extra handy `tf.random_uniform` function. Now how do we use this matrix? Well we want to be able to say that the word "cat" corresponds to the 425th row of the `embed_matrix`, and so first we need to associate each word in the vocabulary with an index. This can be easily achieved with:

```
V = set(corpus)

idx2w = {i:w for (i, w) in enumerate(V)}
w2idx = {w:i for (i, w) in enumerate(V)}
```

Once we can get our corpus into an indexical format, we can then use the other super handy tensorflow function `tf.nn.embed_lookup` to select only the rows from our embedding matrix that we feed it the indexes for. More concretely, for any given batch if we have it in index form:

```
train_inputs = tf.placeholder(tf.float32, shape=[batch_size, num_steps])
train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])

embed_inputs = tf.nn.embed_lookup(embed_matrix, train_inputs)
input = tf.reshape(embed_inputs, (batch_size, num_steps * embed_dim))
```

Now this might look pretty complicated at first, why are we reshaping the embed_inputs variable? And what is a placeholder? 
A tensorflow placeholder is an empty tensor which is defined later on when the graph is being run on actual data, hence why we don't initialize it to any set value. When we train we will "fill" these placeholders with actual values once we can batch our data. So let's talk about the shapes of these matrices and why they are the way that they are. I find it's always easiest to write down the dimensions next to the variable itself to be extra prepared in case something goes wrong. So what is this num_steps variable that we're using? Num_steps refers to the window of _k_ words we want to look at in order to predict the _k_+1th word. Like the bigrams above, if _k_ = 2 then we are predicting the word that follows every pair of 2 words. In practice this is a bottleneck, so try brainstorming some possibile solutions to this! They'll get incorporated to our code in a future post. So if we think of our batch as having 10 examples of _k_ words, we can represent that as a 10 x _k_ matrix of indexes. The next part is going to get a little heavy, so bear with me. `tf.nn.embed_lookup` is a function that takes a tensor and a series of indices to select from that tensor. For example, if we feed `tf.nn.embed_lookup` both `embed_matrix` and `[1, 15, 4]`, it will give us back a matrix where the first row is the 1st row of `embed_matrix`, the 2nd row is the 15th row, and the 3rd is the 4th row. Remember that each row of `embed_matrix` has _d_ dimensions, and so our final matrix will be of the shape _number-of-indices-fed-to-embed_lookup_ x _d_. So now let's think about what happens when you feed `embed_lookup` a matrix of indices rather than a vector. Each row of our index matrix will be treated the exact same as our vector example above, and so each row of our index matrix will result in a 2d matrix. That means our final result will be a 3d matrix where each index row corresponds to a 2d matrix of vectors representing those words. For our purposes this extra dimension will only complicate things, so we can concatenate all of the word vectors in our window of _k_ words together (see figure below). This is what our reshaping step is. We are just concatenating the 2d matrices in our 3d matrix into arrays, resulting in a nice workable 2d matrix. This final matrixes shape will be batch_size x _k_ * _d_. If you don't get that right away don't worry about it, experiment with the data structures for a little bit in a jupyter notebook and see if you can convince yourself that these dimensionalities are what we want. 

The final code for this section looks like:

```
import tensorflow as tf
import numpy as np

V = set(corpus)

idx2w = {i:w for (i, w) in enumerate(V)}
w2idx = {w:i for (i, w) in enumerate(V)}

embed_matrix = tf.Variable(tf.random_uniform([len(V), embed_dim], -1.0, 1.0),name='embed_matrix')

train_inputs = tf.placeholder(tf.float32, shape=[batch_size, num_steps])
train_labels = tf.placeholder(tf.float32, shape=[batch_size, 1])

embed_inputs = tf.nn.embed_lookup(embed_matrix, train_inputs)
input = tf.reshape(embed_inputs, (batch_size, num_steps * embed_dim))
```
Not so bad for 15 or so lines of code!

