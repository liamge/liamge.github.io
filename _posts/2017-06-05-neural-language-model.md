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


