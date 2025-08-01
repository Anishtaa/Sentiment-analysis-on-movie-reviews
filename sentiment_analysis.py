import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import string
import random
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

all_words = [w.lower() for w in movie_reviews.words()]
all_words = [w for w in all_words if w not in stopwords.words('english') and w not in string.punctuation]
all_words_freq = FreqDist(all_words)
word_features = list(all_words_freq)[:2000]

def document_features(document):
    words = set(document)
    return {f'contains({word})': (word in words) for word in word_features}

featuresets = [(document_features(d), c) for (d, c) in documents]

train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

classifier = NaiveBayesClassifier.train(train_set)
print("Accuracy on test set:", accuracy(classifier, test_set))

def predict_sentiment(text):
    words = word_tokenize(text.lower())
    feats = document_features(words)
    return classifier.classify(feats)

# Example
print(predict_sentiment("This movie was thrilling and full of suspense!"))
print(predict_sentiment("The acting was terrible and the plot made no sense."))
