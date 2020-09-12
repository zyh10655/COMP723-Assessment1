from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn import metrics

lemmatizer = WordNetLemmatizer()
hm_lines = 100000


#
# Feature extraction, rather than vector, is equivalent to a method in TF-IDF that imports features as data algorithms.
# This method creates a lexicon from all the words in the corpus.
# It only uses words which occur more than 50 times and less than 1000 times. Effectively it ignores very frequent (such as a, an to from, etc) and very infrequent words.
def create_lexicon(ESE, SSE, MiSE, MoSE, NSE):
    lexicon = []
    with open(ESE, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(SSE, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(MiSE, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(MoSE, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(NSE, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        # print(w_counts[w])
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print("Lexicon size: ", len(l2))
    print("Lexicon: ")
    print(l2)
    return l2


# processes the text for tokenisation and lemmatization
def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r', encoding="utf-8") as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


# Splits the sample into traning and test sets
def create_feature_sets_and_labels(ESE, SSE, MiSE, MoSE, NSE, test_size=0.25):
    lexicon = create_lexicon(ESE, SSE, MiSE, MoSE, NSE)
    features = []
    features += sample_handling('DrugLib_raw/ExtremelySideEffects/ExtremelySideEffects.txt', lexicon, [0, 0, 0, 0, 0])
    features += sample_handling('DrugLib_raw/SevereSideEffects/SevereSideEffects.txt', lexicon, [1, 0, 1, 0, 1])
    features += sample_handling('DrugLib_raw/MildSideEffects/MildSideEffects.txt', lexicon, [0, 1, 0, 1, 0])
    features += sample_handling('DrugLib_raw/ModerateSideEffects/ModerateSideEffects.txt', lexicon, [0, 0, 1, 0, 0])
    features += sample_handling('DrugLib_raw/NoSideEffects/NoSideEffects.txt', lexicon, [1, 1, 1, 1, 1])
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size * len(features))
    # create train and test sets.
    #  # [[5, 8], [7,9]] want all 0th elements
    # [:,0] implies you want all of the 0th elements
    # [5,7] is what you'll get

    # [[[0 1 1 0 1], [0,1]],
    # [features, label ],
    # [features, label]] # features themselves are little one hot arrays
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


#Lexicon size:352(with effectiveness and rating) 343(Without)
# pickle_in = open('assessment1.pickle', 'rb')
# train_x, train_y, test_x, test_y = pickle.load(pickle_in)
'''train a clasifier with the data'''
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(
        'DrugLibT_raw/ExtremelySideEffects/ExtremelySideEffects.txt',
        'DrugLibT_raw/SevereSideEffects/SevereSideEffects.txt',
        'DrugLibT_raw/MildSideEffects/MildSideEffects.txt',
        'DrugLibT_raw/ModerateSideEffects/ModerateSideEffects.txt',
        'DrugLibT_raw/NoSideEffects/NoSideEffects.txt')
    # if you want to pickle this data:
# from sklearn.metrics import classification_report
#
# print(classification_report(test_y, y_pred))
# creates the train and the test sets and serielizes the data and stores it in a binary file
with open('assessment model.pickle', 'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=1, random_state=50)
classifier.fit(train_x, train_y)
y_pred = classifier.predict(test_x)
print("RFC Accuracy:", metrics.accuracy_score(test_y, y_pred))

from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=1)
classifier_knn.fit(train_x, train_y)
y_pred = classifier_knn.predict(test_x)
# Finding accuracy by comparing actual response values(y_test)with predicted response value(y_pred)
print("KNN Accuracy:", metrics.accuracy_score(test_y, y_pred))


'''Print the evaluation metrices'''

from sklearn.metrics import classification_report

print(classification_report(test_y, y_pred))
# creates the train and the test sets and serielizes the data and stores it in a binary file
with open('assessment model.pickle', 'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)
