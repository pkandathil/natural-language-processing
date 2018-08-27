import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

# download_week1_resources()

from grader import Grader
grader = Grader()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')

X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

import re


REPLACE_BY_SPACE_REREPLACE  = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_REREPLACE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)# delete symbols which are in BAD_SYMBOLS_RE from text
    
    # delete stopwords from text
    textWords = text.split()
    resultwords  = [word for word in textWords if word.lower() not in STOPWORDS]
    text = ' '.join(resultwords)
    return text

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)
grader.submit_tag('TextPrepare', text_prepare_results)

X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

# print('-----------------------')
# print(X_train[:3])
# print('-----------------------')

# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}

def count_words(word_list):
  result = {}
  for line in word_list:
    if isinstance(line, list):
        words = line
    else:
        words = line.split( )
    for word in words:
      if(not word in result):
        result[word] = 0
      result[word] += 1
  return result

tags_counts = count_words(y_train)
words_counts = count_words(X_train)

most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags), 
                                                ','.join(word for word, _ in most_common_words)))

##########################################################



DICT_SIZE = 5000
INDEX_TO_WORDS = {}
index = 0
for word in words_counts:
  INDEX_TO_WORDS[index] = word
  index += 1

index = 0
WORDS_TO_INDEX = {}
for word in words_counts:
  WORDS_TO_INDEX[word] = index
  index += 1

ALL_WORDS = WORDS_TO_INDEX.keys()

def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary
        
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.zeros(dict_size)
    words = text.split( )
    for word in words:
      if(word in words_to_index):
        index = words_to_index[word]
        if (index >= dict_size):
          continue
        result_vector[index] = 1
    return result_vector

def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        if (my_bag_of_words(ex, words_to_index, 4) != ans).any():
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_my_bag_of_words())

########################################################
from scipy import sparse as sp_sparse

X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])
X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])
X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])
print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)

#########################################################

row = X_train_mybag[10].toarray()[0]
non_zero_elements_count = 0
for value in row:
  if (value == 0):
    non_zero_elements_count += 1
print('----------------------')
print(non_zero_elements_count)
print('----------------------')
grader.submit_tag('BagOfWords', str(non_zero_elements_count))

##############################################################
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples        
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result
    

    tfidf_vectorizer = TfidfVectorizer (min_df=5, max_df=0.9, ngram_range=(1,2), token_pattern= '(\S+)')
    tfidf_vectorizer = tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
print('X_train_tfidf shape ', X_train_tfidf.shape)
print('X_val_tfidf shape ', X_val_tfidf.shape)
print('X_test_tfidf shape ', X_test_tfidf.shape)
tfidf_reversed_vocab = {i:word for word,i in tfidf_vocab.items()}

contains_c_sharp = False
for word,i in tfidf_vocab.items():  
    if(word == 'c#'):
        contains_c_sharp = True
        break

if(not contains_c_sharp):
    print('------You have a problem with the vectorizer------')
    exit()

################################################################
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier

def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    return OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)

classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)
#########################################################

y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)


y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
####################################################################################
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def print_evaluation_scores(y_val, predicted, predicted_scores):
    print('Accuracy Score: ', accuracy_score(y_val, predicted))
    f1_score_macro = f1_score(y_true=y_val, y_pred=predicted, average='macro')
    f1_score_micro = f1_score(y_true=y_val, y_pred=predicted, average='micro')
    f1_score_weighted = f1_score(y_true=y_val, y_pred=predicted, average='weighted')
    print('F1 score macro', f1_score_macro)
    print('F1 score micro', f1_score_micro)
    print('F1 score weighted', f1_score_weighted)
    precision_score_macro = average_precision_score(y_true=y_val, y_score=predicted_scores, average='macro')
    precision_score_micro = average_precision_score(y_true=y_val, y_score=predicted_scores, average='micro')
    precision_score_weighted = average_precision_score(y_true=y_val, y_score=predicted_scores, average='weighted')
    print('Precision score macro', precision_score_macro)
    print('Precision score micro', precision_score_micro)
    print('Precision score weighted', precision_score_weighted)

print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag, y_val_predicted_scores_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# from metrics import roc_auc
# %matplotlib inline

# n_classes = len(tags_counts)
# roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)

# n_classesn_class  = len(tags_counts)
# roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)

def train_classifier_new(X_train, y_train, penalty_type, coefficient):
    """
      X_train, y_train — training data
      
      return: trained classifier
    """
    
    return OneVsRestClassifier(LogisticRegression(penalty=penalty_type, C=coefficient)).fit(X_train, y_train)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l1', 0.1)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l1 0.1')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l1', 1)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l1 1')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l1', 10)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l1 10')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l1', 100)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l1 100')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l2', 0.1)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l2 0.1')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l2', 1)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l2 1')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l2', 10)
y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
print('Tfidf - l2 10')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

# classifier_tfidf = train_classifier_new(X_train_tfidf, y_train, 'l2', 100)
# y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
# y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
# print('Tfidf - l2 100')
# print_evaluation_scores(y_val, y_val_predicted_labels_tfidf, y_val_predicted_scores_tfidf)

test_predictions = classifier_tfidf.predict(X_test_tfidf)
test_pred_inversed = mlb.inverse_transform(test_predictions)
test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
grader.submit_tag('MultilabelClassification', test_predictions_for_submission)
##################################################

import numpy as np

def find_element_in_list(list_of_items, item): 
    index = -1
    for item_index, list_item in enumerate(list_of_items):
        if(item == list_item):
            index = item_index
            break
    return index

def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
    """
        classifier: trained classifier
        tag: particular tag
        tags_classes: a list of classes names from MultiLabelBinarizer
        index_to_words: index_to_words transformation
        all_words: all words in the dictionary
        
        return nothing, just print top 5 positive and top 5 negative words for current tag
    """
    print('Tag:\t{}'.format(tag))

    
    # Extract an estimator from the classifier for the given tag.
    # Extract feature coefficients from the estimator. 
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    index_of_estimator = find_element_in_list(tags_classes, tag)
    tag_estimator = classifier.estimators_[index_of_estimator]
    tag_coefficients = tag_estimator.coef_[0]
    sorted_coefficients = sorted(tag_coefficients, reverse=True)
    
    highest_coefficients = sorted_coefficients[:5]
    lowest_coefficients = sorted_coefficients[-5:]

    print('-----done------')
    print(highest_coefficients)
    print(lowest_coefficients)
    print('--------')

    top_words = []

    for coeff in highest_coefficients:
        index_of_coeff = find_element_in_list(tag_coefficients, coeff)
        top_words.append(index_to_words[index_of_coeff])
    
    bottom_words = []

    for coeff in lowest_coefficients:
        index_of_coeff = find_element_in_list(tag_coefficients, coeff)
        bottom_words.append(index_to_words[index_of_coeff])

    
    
    top_positive_words = top_words # top-5 words sorted by the coefficiens.
    top_negative_words = bottom_words # bottom-5 words  sorted by the coefficients.
    print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
    print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))

print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)

grader.status()
STUDENT_EMAIL = 'prashantgeorge36@gmail.com' 
STUDENT_TOKEN = 'vI6Ckl6fHsF2AZPe' 
grader.status()
grader.submit(STUDENT_EMAIL, STUDENT_TOKEN)