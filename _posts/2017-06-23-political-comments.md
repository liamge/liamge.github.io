---
layout: post
title: "Conservative vs Liberal Comments on Reddit Part 1: Learning to Tell the Difference"
date: 2017-06-23
comments: true
---

__Disclaimer__: This post will be using the classic bunch of Python scientific libraries (i.e. numpy, sklearn, pandas, etc.). If you're unfamiliar with these, I'd recommend working through the code in here on your own jupyter notebook to really gain a familiarity with these libraries. They're insanely usefull for most of what you'd ever want to do.

# Intro

Before I jump into it, I just want to say that this post isn't intended to speak towards my views of politics, it is simply an interesting dataset to take a look at and a really good playground for some NLP. Of course, there are several key assumptions that are made that may reflect some inherent bias that I have that we will go into below. All that being said, this is meant to be an interesting (and, spoilers, pretty difficult) task that you can cut your NLP teeth in, and hopefully come up with some creative solutions to! 

# Data

So here comes key assumption #1: I got all the data from the r/Conservative and r/Liberal subreddits using the Reddit API wrapper [PRAW](https://praw.readthedocs.io/en/latest). Now of the litany of biases that this introduces, one of these is that r/Conservative has approximately 3x the subscribers as r/Liberal does, introducing a significant difference in the quality of the data that we can get from each subreddit. The smaller size of r/Liberal means that we have to comb further back in time to get a comparable amount of data as we can from r/Conservative, meaning we may simply be modelling a topical difference rather than what we actually want to model: the difference between the way each party uses language to approach a problem. While having an unbalanced dataset is definitely a problem we may have to tackle, it may be worth controlling for the date a comment was posted. Aside from this, we make another potentially deadly assumption: that only conservatives post on r/Conservative and only liberals post on r/Liberal. We can potentially alleviate the severity of this problem by introducing a threshold of the amount of upvotes a coment gets before we include it in the data, relying on the majority of the _voters_ of each subreddit to be the political affiliation of the subreddit they are in, which seems like a much more reasonable assumption. The code for the PRAW scraper is included in the github repository for this post [here](https://github.com/liamge/PoliData/tree/master). For the sake of this post I'll assume you've ran the scraper on both subreddits and so have two csvs that we'll use as our data.

# Exploration

Here is where following along in a jupyter notebook will be the most helpful, as we're going to be cleaning and preprocessing using the style of the sklearn API, writing our own preprocessing transformers that can be used with their Pipeline objects. The best part about this is that it'll give you reusable and portable functions that you can use for most NLP projects, and down the line you can incorporate them into a library you write full of your own tools (which will be covered in a future post).

Let's start by loading in the csvs as a Pandas DataFrame object:

```
import pandas as pd

df_liberal = pd.read_csv('/path/to/my/liberal_data.csv')
df_conservative = pd.read_csv('/path/to/my/conservative_data.csv')
```

We can examine the format of our dataframe with `print(df.head())`, so let's run that on both of our dataframes and see what shape our comments are in. If we've done our data scraping right, we should see that our csvs have two columns, 'text' and 'label'. You'll notice that our labels aren't binarized yet if you used my script. I did that on purpose to show you how sklearn transformers are used (hint: they're magical). We can also see that the 'text' column is in string format with some urls and subreddit mentions, which we'll have to fix if we want to get the text into a computer readable format. Let's combine our two dataframes into a single one containing both conservative and liberal data.

```
dataframes = [df_liberal, df_conservative]
df = pd.concat(dataframes)
```

Let's see what our distribution of labels is. (Note: if you're using a jupyter notebook to run this, make sure you run `%matplotlib inline` first so you can see the plot within the notebook.)

```
from collections import Counter

counts = Counter(df['label'].values)
tmp = pd.DataFrame.from_dict(counts, orient='index')
tmp.plot(kind='bar')
```

All that extra syntax there is so we can count these labels in their current format. We really should be binarizing them, which I will walk you through below. If you used my scraper you should see that there's an imbalance of classes here, with conservative occuring more frequently than liberal. This confirms our suspicions from before. It's not a disaster though, we just simply have to keep in in mind when we build our model. Speaking of, let's jump right into preprocessing.

# Preprocessing

If you've never used sklearn's transformers, essentially they are objects that have both a `fit` and `transform` method in them which transforms your data into a new format. More concretely, say we wanted to binarize our labels so our algorithm can properly use them. We would use sklearn's `LabelBinarizer` transformer like so:

```
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

y = lb.fit_transform(df.copy()['label'].values)
y = y.reshape(y.shape[0],)

print(y[:10])
```

We can see that the variable `y` now contains the binarized form of our 'label' column in the dataframe. Pretty easy stuff, but now we can move on to the 'text' column. Notice the `reshape` line; that will help when we use cross validation in sklearn.

This column poses a couple interesting challenges to us, the first of us is how do we want our text formatted? In NLP there are a ton of different ways to represent text, but they all start with simple tokenization. Sklearn provites several options for text transformers, but sometimes they take care of too many stages in preprocessing at once. For example, sklearn's `CountVectorizer` tokenizes the data and transforms it into what's called a "one hot" representation. A "one hot" representation assigns each unique word in the corpus's vocabulary, V,  an index, and then represents a sentence as a vector of |V| dimensionality, where the i'th index of that vector is the count of the word with the i'th index in that sentence. While super convenient if you're going to use that form of representation, it doesn't give us just the tokenized form of the text if we need that for an alternative representation (which we will in a separate post).

So let's write our own simple Tokenizer transformer such that it can be used exactly like other sklearn transformers. In order to do so, we need to create on object, `Tokenizer`, that inherets properties from sklearn's BaseEstimator and TransformerMixin classes. All we need to do is write `fit()` and `transform()` methods in our class, and inheriting from TransformerMixin gives us `fit_transform()` for free. Similarly, BaseEstimator gives us `get_params()` and `set_params()`, which are both useful for hyperparameter tuning. (All of this information I learned from Aurelien Geron's (no relation) fantastic book [Hands-On Machine Learning with Scikit-Learn and Tensorflow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291))

```
from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        # Nothing here
        return self

    def transform(self, X, y=None):
        from nltk import word_tokenize
        tokenized = []
        for text in X:
            tokenized.append(word_tokenize(text))

        return tokenized
```

Now we can get our tokenized text like so:

```
tk = Tokenizer()

tokenized = tk.fit_transform(df.copy()['text'].values)
```

We will use that format of creating our own custom transformers in a future post when we start moving towards more complex solutions, but for our quick analysis right now lets use a simple [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) to represent our text so we can see how our models will do on this task.

```
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

text_pipeline = Pipeline([
	('counts', CountVectorizer()),
	('tfidf', TfidfTransformer())
])

X = text_pipeline.fit_transform(df.copy()['text'])
```

Our variable `X` should now be a scipy sparse matrix representing our data. If you're unfamiliar with what [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) is, it essentially weights words based upon their relative frequency, theoretically weighting more "important" words higher than less "important" words.

Before we inspect the data too much, let's break this up into train and test data so we don't cheat.

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.reshape(y_train.shape[0],)
y_test = y_test.reshape(y_test.shape[0],)
```

The random_state argument will allow for consistency of the random shuffling of the data.

With this data we can already jump straight to cursory model building, which sklearn makes insanely easy.

# Basic Model Building

For this quick drafting let's run a couple different models and select a couple to do some hyperparameter tuning with. Let's start with a couple simple options.

```
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, cross_val_score

scores = []

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for classifier in classifiers:
    score = cross_val_score(classifier, X_train.toarray(), y_train, cv=3, scoring='f1')
    scores.append(score.mean())

print(list(zip(names, scores)))
```

Ok, maybe more than just a couple. Warning, this block of code is going to take a long time to run because you're training 10 different models on the data 3 times each (due to cross validation). It may be advisable to reduce the dimensionality of the data via something like Truncated SVD (as opposed to PCA because we have sparse input, plus if we perform SVD on a TDIFT weight matrix we get [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)).

Just replace the `for` loop with this:

```
from sklearn.decomposition import TruncatedSVD

for classifier in classifiers:
    pipeline = Pipeline([
                        ('svd', TruncatedSVD(n_components=100)),
                        ('clf', classifier)
               ])
    score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='f1')
    scores.append(score.mean())

print(list(zip(names, scores)))
```

Once you select a couple promising looking models (which we'll get to in a second, because it's more complicated than it sounds), you can begin hyperparameter tuning with sklearn's `GridSearchCV`, which tries every hyperparameter combination you give it and uses cross validation to assess how well that model performs, or sklearn's `RandomizedSearchCV` which does the same sort of tuning but from random samples for `n` iterations. Something to keep in mind is that GridSearchCV actually runs each hyperparameter combination 3 times (for cross validation), and so will take quite a while if you do an exhaustive search whereas you can control exactly how many times `RandomizedSearchCV` runs.

Before we get into all that, let's inspect our promising models and see what kind of mistakes they make. If your data is anything like mine, after SVD Naive Bayes and QDA looked to be the most promising models. This makes sense, Naive Bayes is known to be a pretty good baseline for most text based tasks. Let's take more of a deep dive into their predictions.

```
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

nb_pipeline = Pipeline([
    ('svd', TruncatedSVD(n_components=100)),
    ('clf', GaussianNB())
])

qda_pipeline = Pipeline([
    ('svd', TruncatedSVD(n_components=100)),
    ('clf', QuadraticDiscriminantAnalysis())
])

nb_preds = cross_val_predict(nb_pipeline, X_train, y_train)
qda_preds = cross_val_predict(qda_pipeline, X_train, y_train)

nb_mat = confusion_matrix(nb_preds, y_train)
qda_mat = confusion_matrix(qda_preds, y_train)

plt.matshow(nb_mat)
plt.ylabel('predicted')
plt.xlabel('actual')
plt.title('NB Matrix')

plt.matshow(qda_mat)
plt.ylabel('predicted')
plt.xlabel('actual')
plt.title('QDA Matrix')
```

Again, if your data is like mine you'll see that they each have particular strengths and weaknesses. QDA seems to be much better at predicting true conservatives, but the tradeoff is that it seems to predict that most things are conservative. This is the _precision_ vs. _recall_ tradeoff. QDA seems to have very good _recall_, but pretty awful _precision_. We may be able to leverage both model's strengths and aleviate their weaknesses by ensembling them. In order to do that, let's inspect some other promising models. In particular, some more powerful models can be useful with some proper hyperparameter tuning and fidgeting despite their current performance. Let's focus in on [Extreme Gradient Boosting](http://xgboost.readthedocs.io/en/latest/model.html), an extremely powerful model that seems to be performing pretty poorly. First, let's see how it predicts on our dataset as is.

```
xgb_clf = XGBClassifier()

preds = cross_val_predict(xgb_clf, X_train, y_train)

mat = confusion_matrix(y_train, preds)

plt.matshow(mat)
plt.ylabel('predicted')
plt.xlabel('actual')
plt.title('XGB Matrix')

print("F1 score: {}".format(f1_score(preds, y_train)))
```

Not great. It actually almost never predicts that a comment is made by a liberal. One advantage to using XGB is it can give us an estimate of the feature importances. Let's visualize them:

```
clf = XGBClassifier().fit(X_train, y_train)

word_idx = list(text_pipeline.named_steps['counts'].vocabulary_.items())
sorted_words = [w[0] for w in sorted(word_idx, key=lambda x: x[1])]

importances = clf.feature_importances_

sorted_importances = sorted(list(zip(sorted_words, importances)), key=lambda x: x[1], reverse=True)

print(sorted_importances[:50])
```

Makes a lot of sense, these are pretty charged and polarizing words. That's pretty encouraging that our model can pick up on all of this from just word counts! 

Now let's see what ensembling our three models gets us. It may alleviate this underprediction problem.

```
from sklearn.ensemble import VotingClassifier

clf1 = Pipeline([
    ('svd', TruncatedSVD(n_components=100)),
    ('clf', GaussianNB())
])
clf2 = Pipeline([
    ('svd', TruncatedSVD(n_components=100)),
    ('clf', QuadraticDiscriminantAnalysis())
])
clf3 = XGBClassifier()

eclf = VotingClassifier(estimators=[('nb', clf1), ('qda', clf2), ('xgb', clf3)], voting='soft')

preds = cross_val_predict(eclf, X_train, y_train)

mat = confusion_matrix(y_train, preds)

plt.matshow(mat)

print("F1 score ensemble: {}".format(f1_score(preds, y_train)))
```

Pretty good actually! Although the XGB classifier by itself managed to do a little better, I have a feeling this will generalize a little more to the testing set.

We managed to get some decent performance from a couple simple models and one complicated one. If we wanted to really push our performance, we would at this stage start using hyperparameter tuning to really push our performance. I'm going to get you started, though most of this stuff really just comes down to your willingness to stare at a training model for hours at a time.

Initially I'd suggest starting with sklearn's `RandomizedSearchCV` to get reasonable values for your hyperparameters, and then moving on to `GridSearchCV` to really fine tune it. Sklearn has a great functionality with doing this for ensemble techniques:

```
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint

param_dist = {"nb__svd__n_components": sp_randint(2, 1000),
              "qda__svd__n_components": sp_randint(2, 1000),
              "xgb__svd__n_components": sp_randint(2, 1000),
              "xgb__clf__max_depth": sp_randint(3, 10)}

n_iter_search = 20
random_search = RandomizedSearchCV(eclf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(X_train, y_train)
```

Warning, this is going to take a really long time to run since it's training the entire ensemble a huge amount of times! Much like this syntax, you can fine tune more precisely with `GridSearchCV`, but I'll leave that as an exercise to the reader. Note that we aren't tuning a huge amount of hyperparameters because I have a feeling that this isn't going to be our final solution (*cough* word vectors *cough*) for this dataset, so I don't feel like investing the huge compute time required for hyperparameter tuning.

Finally, let's see how we do with our classifier now on our testing set!

``` 
from sklearn.metrics import precision_score, recall_score

print("Accuracy: {}".format(accuracy_score(preds, y_test)))
print("Precision: {}".format(precision_score(preds, y_test)))
print("Recall: {}".format(recall_score(preds, y_test)))
print("F1: {}".format(f1_score(preds, y_test)))
print(confusion_matrix(y_test, preds))
```

Not terrible, but certainly not great by any stretch of the imagination. It still has trouble accurately predicting when a comment is liberal.

# Final Notes

In this post we went through a good cursory Data Science pipeline, from data mining to basic model building/evaluation, to finally building one huge ensemble method and performing some hyperparameter tuning on it. Hopefully you got a sense of the methodology used in a standard pipeline, and you got a good appreciation for the amazing tools that sklearn has to offer. Transformers, Pipelines, Cross Validation, all of these things sklearn makes incredibly easy and cheap to perform. In the next post we will take this same dataset and come up with a more Deep Learning oriented solution, examining word vectors and how those can be used for a continuous representation of the Reddit comments. We will use these representations later on when we build our final model, leveraging Deep Learning, word vectors, and all we've learned in this post re: sklearn's API and combining all of these features into one unified model that we can deploy. As always, I encourage you to leave any comments/suggestions down below!
