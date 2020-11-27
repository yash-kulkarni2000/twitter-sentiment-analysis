#Importing the libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Importing the training set and splitting it as required
old_training_set = pd.read_csv('train.csv')
training_set = old_training_set[['selected_text', 'sentiment']]

#Label Encoding the dependent variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
training_set.iloc[:, 1] = le.fit_transform( training_set.iloc[:, 1] )

#Selected text on which we have to train the data
sel_text = training_set.iloc[:, 0]

#Count Vectorizer for tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform( training_set['selected_text'].values.astype('U') ).toarray()
y = training_set.iloc[:, 1].values

#Train Test Split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Using Naive Bayes Classifier and training it 
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

#Predicting and visualizing results next to each other 
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

#Using model to train it on another test 
test_set = pd.read_csv('submission.csv')
test_set_final = cv.fit_transform( test_set['selected_text'].values.astype('U') ).toarray()
y_pred_new = classifier.predict(test_set_final)

#To cross-check the predictions
submission = pd.read_csv('test.csv')
submission.iloc[:, 2] = le.fit_transform( submission.iloc[:, 2] )
sentiment_og = submission.iloc[:, 2]
accuracy_score(sentiment_og, y_pred_new)



'''
from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0]}
grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best accuracy: ", (best_accuracy * 100))
print("Best parameters: ", best_parameters )
Best accuracy:  76.6966885007278
Best parameters:  {'alpha': 0.1}
'''

