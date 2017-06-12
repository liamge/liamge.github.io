---
layout: post
title: "Using Data Science in the Wild: Exploring the Bachelorette"
date: 2017-06-12
comments: true
---

__DISCLAIMER__: You don't have to watch the Bachelorette to properly experience this post, but you should watch the bachelorette to properly experience life.

# Intro

I have just recenlty started the whole armada of "Bachelor" brand television shows (which are amazing flaming garbage pieces of television) and it ocurred to me before the preview of this most recent season that they present a rather unique and interesting Data Science challenge: can we predict who the producers will like just based off of the features that are publically available for them? So I began to dig around and found that there aren't very good public datasets available of old Bachelorette/Bachelor contestants. So I made one.

## Working with the data

First I had to gather the data which I extracted from each of the Bachelorette season's Wikipedia articles [here](https://en.wikipedia.org/wiki/The_Bachelorette). This data contains each contestants name, age, hometown, occupation, and what week, if at all, they were eliminated. I went ahead and manually labelled if the datapoint was a Bachelorette. This isn't exactly the best data, it's mostly categorical and name is already pretty much useless as a predictor unless we get _really_ creative with the feature engineering, which we can always try later. At any rate, let's dive into it and see if we can't figure out some good features to use (all the code will be gathered [here]() if you want to take a look at it).

First off, download the dataset [here]() so you can follow along in a notebook. Let's just start by importing our libraries that we're going to be using for the data manipulation.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

And if you are running this in a Jupyter notebook (which if you aren't you _really_ should try them out, they are amazing) you can add `%matplotlib inline` in your cell to allow for plotting to display in the notebook itself. Now let's load our data into a Pandas DataFrame object and print the head:

```
df = pd.read_csv('bachelorette.csv')

print(df.head())
```

You can see the output is as follows:

```
   Unnamed: 0  Age                             Hometown  Label  \
0           0   30                Indianapolis, Indiana    0.0   
1           1   30                    Beaverton, Oregon    0.0   
2           2   30                         Mentor, Ohio    0.0   
3           3   30                    Marietta, Georgia    0.0   
4           4   29  Vancouver, British Columbia, Canada    0.0   

                     Name          Occupation  Season  Week IsBachelorette  
0             Trista Rehn  Physical therapist       1   1.0           True  
1       Meredith Phillips       Makeup artist       2   1.0           True  
2  Jennifer "Jen" Schefft           Publicist       3   1.0           True  
3           DeAnna Pappas   Real estate agent       4   1.0           True  
4          Jillian Harris   Interior designer       5   1.0           True  
```

So as you can see it looks like the first few datapoints are our Bachelorettes. We can inspect them manually (or know via encyclopedic knowledge of the Bachelorette) to see that there are 13 Bachelorettes that we are working with, though we would like to separate out the 13th season now as that is the season we are trying to predict.

```
train_df = df[df['Season'] != 13]
test_df = df[df['Season'] == 13]
```

Let's inspect our training data with `df.hist()`:

![](/assets/train_df_graphs.png)

And be sure to check the correlation matrix with `df.corr()`.

As we can see, very few of the features are actually useful to us in it's current form. Age seems relatively normally distributed, and season seems to be uniformly distributed more or less. Week is majorly right skewed, which makes sense as more people go home on the first week than on any other week (also I filled all NAs with 1 so it thinks all the Bachelorettes go home in the first week, so you can fix that if you want a more accurate graph). Also we can get a glimpse of what's probably going to be a major problem; there is a huge class imbalance in our data! It makes sense, only one person per season ever actually wins, so we're going to have to come up with a solution to this problem down the line. 

Let's use the age of the contestant and compare it to the age of the Bachelorette and see where that gets us.
