import pandas as pd
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


train_path = '/Users/ghazalghalebandi/PycharmProjects/pdf-parser/train.csv'
train = pd.read_csv(train_path, header='infer')
print(train.head())
print('row and col ', train.shape)
print(train.groupby('label').count())


test_path = '/Users/ghazalghalebandi/PycharmProjects/pdf-parser/test.csv'
test = pd.read_csv(test_path, header='infer')
print(test.head())
print('row and col ', test.shape)
print(test.groupby('label').count())

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_label = encoder.fit_transform(train['label'])
test_label = encoder.fit_transform(test['label'])




# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train['text'])

xtrain_tfidf =  tfidf_vect.transform(train['text'])
xtest_tfidf =  tfidf_vect.transform(test['text'])


lr = LogisticRegression(solver='lbfgs')
lr.fit(xtrain_tfidf,train_label)
predictions = lr.predict(xtest_tfidf)
print(predictions)


from sklearn.metrics import classification_report
print(classification_report(test['label'], predictions))

# with 5000 max features

#               precision    recall  f1-score   support
#
#            0       1.00      0.02      0.03       506
#            1       0.71      1.00      0.83      1204
#
#     accuracy                           0.71      1710
#    macro avg       0.85      0.51      0.43      1710
# weighted avg       0.79      0.71      0.59      1710




