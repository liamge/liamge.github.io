---
layout: post
title: "Learning to Implement Papers: Neural Language Model in Tensorflow from Scratch"
date: 2017-06-05
comments: true
---

Learning how to implement a paper is a skill often ignored by people first starting out in NLP, but it is this skill in which one of the core tenants of science takes place: reproducibility. Learning to implement a paper will not only help you as a programer and a researcher, but it will help the science progress as well. Implementing a paper can be intimidating (dense academic text doesn't exactly translate easily to code) but sitting down with a paper for a while and really going through the process of parsing exactly what needs to happen where is the best way to learn about a topic. For this post I will be implementing the seminal paper from [Yoshua Bengio et al., 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), which introduced the possibility of using a Neural Network to create a highly accurate Language Model. The use of Neural Networks for Language Modeling (which will be explained in the next section, don't worry) is now the state of the art in Language Modeling.

## Language Models

What exactly is the inscrutably named Language Model? A [Language Model](https://en.wikipedia.org/wiki/Language_model) is a probability distribution over sequences of words. If you don't have a statistics/probability background however, this won't mean anything at all, so let's focus on the intuition a bit before we jump into the math of it. Say you were trying to translate a sentence from French into English. The French sentence could be something like: "J'aime bien la PNL!". The problem with translating this is that there are a huge number of ways that one _could_ translate it, for example: "I love NLP!" or "I really like NLP!" or even "I like well the NLP!". What we need to be able to do is take all these possible translations and ask which one sounds the most "English". That's where the Language Model comes in. The ideal Language Model would be able to tell you that "I love NLP!" is way more "English" than "I like well the NLP!", despite the latter having a closer correspondance with the actual French words.

Great, so now that we know what a Language Model _does_ we need to ask ourselves _how_ a Language Model does it. The old tried and true methods are called _N_-gram Language Models, and they actually are pretty much always a competitive baseline. Essentially the way that they work is they take an _n_-gram (an ordered tuple of _n_ words) and they estimate the probability of a sequence of words as the product of the probabilities of all of it's constituent _n_-grams. More intuitively, if you are given the sentence: "The cat is pretty neat", you can break it down into _bigrams_ (_n_-grams with 2 items) which would look like this: (The, cat), (cat, is), (is, pretty), (pretty, neat). From here, the probability of the entire string is just the product of the probability of each bigram, which is calculated from some larger corpus of data as: 

` # of times W2 follows W1 / total # of times W1 is used `

or intuitively, the relative frequency that W2 is used after W1. In practice, these _n_-gram Language Models have some significant drawbacks: the higher _n_ is the more memory they use, and the larger corpus you use the larger your vocabulary of possible _n_-grams gets.
