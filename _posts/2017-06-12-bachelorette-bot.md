---
layout: post
title: "Using Data Science in the Wild: Exploring the Bachelorette"
date: 2017-06-12
comments: true
---

DISCLAIMER: You don't have to watch the Bachelorette to properly experience this post, but you should watch the bachelorette to properly experience life.

# Intro

I have just recenlty started the whole armada of "Bachelor" brand television shows (which are amazing flaming garbage pieces of television) and it ocurred to me before the preview of this most recent season that they present a rather unique and interesting Data Science challenge: can we predict who the producers will like just based off of the features that are publically available for them? So I began to dig around and found that there aren't very good public datasets available of old Bachelorette/Bachelor contestants. So I made one.

## Gathering the data

First I had to gather the data which I extracted from each of the Bachelorette season's Wikipedia articles [here](https://en.wikipedia.org/wiki/The_Bachelorette). This data contains each contestants name, age, hometown, occupation, and what week, if at all, they were eliminated.
