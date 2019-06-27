#author: Nazim Bahadir BARDAKCI nazimbahadirbardakci[at]gmail[dot]com
#import requirement libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, f1_score, recall_score
import numpy as np
from sklearn import preprocessing

pdata = pd.read_csv("input.csv")
print(pdata.head())
print(pdata.describe())
kf = KFold(n_splits=10)
kf.get_n_splits(pdata)
# First - split into Train/Test

features = list(pdata.columns.values)
features.remove('result')
print(features)
X = pdata[features]
y = pdata['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Decision Tree
from sklearn import tree, preprocessing

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
print("Decision Tree score : ",clf.score(X_test,y_test))
print("DecisionTree score cros_val_score  : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#Decision Tree V2
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    random_state=0)
scores = cross_val_score(clf, X, y, cv=kf)
print("sklearn.tree lib Decision Tree score  cros_val_score  : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
#Random Forest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                             random_state=0)
clf = clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
print("Random Forest score : ",clf.score(X_test, y_test))
print("Random Forest score cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#KNN
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

nca_pipe = Pipeline([('knn', knn)])
nca_pipe.fit(X_train, y_train)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
print("KNN result score : ",nca_pipe.score(X_test, y_test))
print("KNN cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#SVM
from sklearn import svm

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
clf.predict(X_test)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
# get support vectors
clf.support_vectors_
# get indices of support vectors
clf.support_
#get number of support vectors for each class
clf.n_support_
print("SVM result score : ",clf.score(X_test, y_test))
print("SVM cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
#NeuralNetwork(SuperVised)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(8, 4), random_state=1)

clf.fit(X_train, y_train)
scores = cross_val_score(clf, X, y, cv=kf)
print("Neural Network score : ",clf.score(X_test, y_test))
print("Neural Network  cros_val_score : ",scores.mean())
y_pred = clf.predict(X_test)
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#GradientBoostingClassifier
from sklearn import ensemble

clf =ensemble.GradientBoostingClassifier(n_estimators=265,
                                               validation_fraction=0.2,
                                               n_iter_no_change=5, tol=0.01,
                                               random_state=0)
clf = clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
y_pred = clf.predict(X_test)
print("Gradient Tree Boosting Classifier score : ",clf.score(X_test, y_test))
print("Gradient Tree Boosting Classifier cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#Ensemble
from sklearn.ensemble import  ExtraTreesClassifier

clf = ExtraTreesClassifier(max_depth=3, n_estimators=10, random_state=0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
y_pred = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
print("Extra Trees Classifier score : ",clf.score(X_test, y_test))
print("Extra Trees Classifier cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)

y_pred = clf.predict(X_test)
print("Random Forest score : ",clf.score(X_test, y_test))
print("Random Forest cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#Naive Bayes
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
y_pred = clf.predict(X_test)
print("Naive Bayes score : ",clf.score(X_test, y_test))
print("Naive Bayes cros_val_score : ",scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-0)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=kf)
y_pred = clf.predict(X_test)
print("Stochastic Gradient Descent score : ",clf.score(X_test, y_test))
print("Stochastic Gradient Descent cros_val_score : ", scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#Gradient Boosting
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}
clf = ensemble.GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = clf.score(X_test, y_test)
scores = cross_val_score(clf, X, y, cv=kf)
print("Gradient Boosting score : ", acc)
print("Gradient Boosting Score : ", scores.mean())
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))


#Keras
print(X_train.shape)
print(X_test.shape)
NB_EPOCHS = 265  # num of epochs to test for
BATCH_SIZE = 1326
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 265)
classifier.fit(X_train, y_train)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = kf)
y_pred = clf.predict(X_test)
print ("Keras cros_val_score : ", accuracies.mean()*100)
print("Keras result : ", classifier.score(X_test,y_test))
print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
variance = accuracies.std()
print("Accuracy variance: "+ str(variance))
predictions = classifier.predict(X_test)
predictions = predictions.reshape(-1)
df = pd.DataFrame()
y_test = y_test.reset_index(drop=True)
df["predictions"] = predictions
df["labels"] = y_test
print(df)
