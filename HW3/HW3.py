# Tim Zhang
# 110746199
# CSE537 HW 3
#---------------------------------------------------
import sys
import sklearn, sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import re
import os, os.path
import shutil
import codecs
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
from nltk.stem import *
import pickle

#---------------------------------------------------
# SOURCE: https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/datasets/twenty_newsgroups.py#L154
# I was unable to correctly import these functions via modules
#
def strip_newsgroup_header(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    return after
#---------------------------------------------------

#
# Function loads the data set for learning
#
def load_data(path):
	categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'talk.religion.misc']
	data = sklearn.datasets.load_files(path, categories=categories, encoding="utf-8", decode_error="replace", shuffle=True, random_state=42)

	return data

#
# Removes header, footer, and quotes from data set
# The new files are written into a directory "preprocessed"
#
def preprocess_data(path):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	preprocessed_path = dir_path + '/preprocessed' + path

	if not os.path.exists(preprocessed_path):
		shutil.copytree(dir_path + path, preprocessed_path)
	else: 
		return
	
	for dirpath, dirnames, files in os.walk(preprocessed_path):
	    for name in files:
	        f = codecs.open(os.path.join(dirpath, name), 'r+', 'utf8', 'ignore')
	        text = strip_newsgroup_header(f.read())
	        f.seek(0)
	        f.write(text)
	        f.truncate()
	        f.close()

#
# Classification pipeline method
#
def classify(train, test, classifier, list):
	global f1_scores
	training = copy.deepcopy(train)

	text_clf = classifier

	text_clf = text_clf.fit(training.data, training.target)
	predicted = text_clf.predict(test.data)

	if sys.argv[1] != '1':
		print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
		print("Macro Average F1: " + str(metrics.f1_score(test.target, predicted, average='macro')) + "\n")

	list.append(metrics.f1_score(test.target, predicted, average='macro'))

#
# Runs each classifier with specified ngram
#
def compareAlgorithms(ngram, training, test):
	global f1_scores

	# Naive Bayes
	if sys.argv[1] == '0':
		print("Naive Bayes")

	classify(training, test, Pipeline([('vect', CountVectorizer(ngram_range=ngram)), 
									   ('tfidf', TfidfTransformer()), 
									   ('clf', MultinomialNB())]), f1_scores[0])

	# Logistic Regression
	if sys.argv[1] == '0':
		print("Logistic Regression")
	
	classify(training, test, Pipeline([('vect', CountVectorizer(ngram_range=ngram)), 
									   ('tfidf', TfidfTransformer()),
									   ('clf', LogisticRegression())]), f1_scores[1])

	# SVM
	if sys.argv[1] == '0':
		print("SVM")
	
	classify(training, test, Pipeline([('vect', CountVectorizer(ngram_range=ngram)), 
									   ('tfidf', TfidfTransformer()), 
									   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=100))]), f1_scores[2])

	# Random Forest
	if sys.argv[1] == '0':	
		print("Random Forrest")
	
	classify(training, test, Pipeline([('vect', CountVectorizer(ngram_range=ngram)), 
									   ('tfidf', TfidfTransformer()),
									   ('clf', RandomForestClassifier())]), f1_scores[3])

#
# Compares learning rates of each classification algorithm with various sizes of training data
#
def compareLearningCurve():
	splitSize = []

	print("Learning Curve")

	for i in range(10):
		split = splitData(training, test, float(i + 1) / 10, splitSize)
		compareAlgorithms((1, 1), split, test)

	for classifier in range(4):
		plt.plot(splitSize, f1_scores[classifier])
		plt.xlabel('Training Size')
		plt.ylabel('F1-Score')
		plt.axis([0, splitSize[9], 0, 1.0])
		plt.savefig('classifier' + str(classifier) + '.png')

#
# Splits training data into specified percentage and then tests each algorithms performance
#
def splitData(training, test, percentage, size):
	split = copy.deepcopy(training)

	if percentage == 1:
		percentage = .9999999  # Handles train_test_split arguement expectations

	splitTestData, split.data, splitTestTarget, split.target = train_test_split(training.data, training.target, test_size=percentage)

	size.append(len(split.data))

	return split

#
# Exploration results using 8 different configurations on SVM
#
def explorationResults(training, test):
	# Process training data
	no_stopwords = removeStopwords(training)
	stemmed = stem(training)
	stemmed_no_stopwords = stem(removeStopwords(training))
	
	# Process test data
	no_stopwords_test = removeStopwords(test)
	stemmed_test = stem(test)
	stemmed_no_stopwords_test = stem(removeStopwords(test))

	# Unigram Baseline
	print("Unigram Baseline")
	classify(training, test, 
		     Pipeline([('vect', CountVectorizer()), 
		               ('tfidf', TfidfTransformer()), 
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Removed Stopwords Training Data
	print("Removed Stopwords")
	classify(no_stopwords, no_stopwords_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])	

	# Stemmed Training Data
	print("Porter Stemmed")
	classify(stemmed, stemmed_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Removed Stopwords + Stemmed Training Data
	print("Removed Stopwords + Porter Stemmed")
	classify(stemmed_no_stopwords, stemmed_no_stopwords_test, 
		     Pipeline([('vect', CountVectorizer()), 
			           ('tfidf', TfidfTransformer()), 
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Univariate Feature Selection
	print("Univariate Feature Selection")
	classify(training, test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectPercentile(chi2, 25)),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# L2 Regularization
	print("L2 Regularization")
	t = copy.copy(training)
	classify(training, test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectFromModel(LinearSVC(penalty="l2"))),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# L2 Regularization + Univariate Feature Selection
	print("L2 Regularization + Univariate Feature Selection")
	t = copy.copy(training)
	classify(training, test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('univariate', SelectFromModel(LinearSVC(penalty="l2"))),
					   ('L2', SelectPercentile(chi2, 25)),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Univariate Feature Selection + L2 Regularization
	print("Univariate Feature Selection + L2 Regularization")
	t = copy.copy(training)
	classify(training, test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('L2', SelectPercentile(chi2, 25)),
					   ('univariate', SelectFromModel(LinearSVC(penalty="l2"))),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Removed Stopwords + Univariate Feature Selection
	print("Removed Stopwords + Univariate Feature Selection")
	classify(no_stopwords, no_stopwords_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectPercentile(chi2, 25)),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Removed Stopwords + L2 Regularization
	print("Removed Stopwords + L2 Regularization")
	classify(no_stopwords, no_stopwords_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectFromModel(LinearSVC(penalty="l2"))),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Stemmed Training Data + Univariate Feature Selection
	print("Porter Stemmed + Univariate Feature Selection")
	classify(stemmed, stemmed_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectPercentile(chi2, 25)),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Stemmed Training Data + L2 Regularization
	print("Porter Stemmed + L2 Regularization")
	classify(stemmed, stemmed_test, 
		     Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectFromModel(LinearSVC(penalty="l2"))),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

	# Stemmed Training Data + Removed Stopwords + Univariate Feature Selection
	print("Porter Stemmed + Removed Stopwords + Univariate Feature Selection")
	classify(stemmed_no_stopwords, stemmed_no_stopwords_test, 
			 Pipeline([('vect', CountVectorizer()), 
					   ('tfidf', TfidfTransformer()), 
					   ('selector', SelectPercentile(chi2, 25)),
					   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))]), [])

#
# Stems data set using Porter stemmer
#
def stem(data):
	stemmer = PorterStemmer()
	stemmed = copy.deepcopy(data)

	for i in range(len(stemmed.data)):
		stemmedWords = [stemmer.stem(word) for word in stemmed.data[i].split()]
		stemmed.data[i] = ' '.join(stemmedWords)

	return stemmed

#
# Removes stop words as a preprocessing stop
#
def removeStopwords(data):
	no_stopwords = copy.deepcopy(data)
	# Stopwords taken from http://www.nltk.org/book/ch02.html
	stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
				 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
				 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
				 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
				 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
				 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
				 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
				 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
				 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
				 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
				 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
				 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

	for i in range(len(no_stopwords.data)):
		removed = [word for word in no_stopwords.data[i].split() if not word.lower() in stopwords]
		no_stopwords.data[i] = ' '.join(removed)

	return no_stopwords

#
# Trains "My Best Configuration" on training set
#
def trainModel(training, name):
	stemmed_no_stopwords = stem(removeStopwords(training))

	# Removed Stopwords + Stemmed Training Data
	print("Removed Stopwords + Porter Stemmed")

	classifier = Pipeline([('vect', CountVectorizer()), 
			  			   ('tfidf', TfidfTransformer()), 
						   ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

	classifier = classifier.fit(stemmed_no_stopwords.data, stemmed_no_stopwords.target)

	# Save the classifier
	with open(name, 'wb') as fid:
		pickle.dump(classifier, fid)

#
# Runs "My Best Configuration" on test set
#
def useModel(modelPath, test):
	stemmed_no_stopwords_test = stem(removeStopwords(test))

	with open(modelPath, 'rb') as fid:
	    model = pickle.load(fid)

	predicted = model.predict(stemmed_no_stopwords_test.data)

	print("My Best Configuration")

	print(metrics.classification_report(stemmed_no_stopwords_test.target, predicted, target_names=stemmed_no_stopwords_test.target_names))
	print("Macro Average F1: " + str(metrics.f1_score(stemmed_no_stopwords_test.target, predicted, average='macro')) + "\n")

#
# Main
#
f1_scores = [[] for x in range(4)]  # Holds all computed F1-scores, used for learning curve

# Get training data
if sys.argv[1] != '4':
	training_file_path = sys.argv[2]
	preprocess_data(training_file_path)
	training = load_data('preprocessed' + training_file_path)

# Get test data
if sys.argv[1] != '3':
	test_file_path = sys.argv[3]
	preprocess_data(test_file_path)
	test = load_data('preprocessed' + test_file_path)

# Compare learning algorithms
if sys.argv[1] == '0':
	print("Unigram Baseline")
	compareAlgorithms((1, 1), training, test)  # Unigram

	print("Bigram Baseline")
	compareAlgorithms((2, 2), training, test)  # Bigram

# Compare learning rate
elif sys.argv[1] == '1':
	# Run learning curve algorithm with 10% split increments
	compareLearningCurve()

# Exploration results
elif sys.argv[1] == '2':
	explorationResults(training, test)

# Train MBC, then save model to disk
elif sys.argv[1] == '3':
	trainModel(training, sys.argv[3])

# Run stored MBC model on test set
else:
	useModel(sys.argv[2], test)