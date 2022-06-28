from numpy import mean
import matplotlib.pyplot as plt
from train_classifier import prepTrainingData
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

#Function to do cross validation
def crossValidationNKFold(X, y, model, test_size, K):

    classifier, X, y = prepTrainingData(X, y, test_size, model)
    # prepare for cross validation
    cross_vali = KFold(n_splits=K, random_state=1, shuffle=True)
    # evaluate model
    scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cross_vali, n_jobs=-1)
    # report on performance
    print('accuracy scores:', scores)
    print('average accuracy: %.3f' % (mean(scores)))


#Function to print out the accuracy scores and classification result
def reportAndScore(X, y, model, test_size):
    # train the Random Forest classifier
    Classifier, X_test, y_test = prepTrainingData(X, y, test_size, model)
    # make prediction on the test set X
    prediction = Classifier.predict(X_test)
    # print out the classification report of test set y and the prediction 
    # result from test set X
    if model == 'NN':
        print(classification_report(y_test, prediction, zero_division=0))
    else:
        print(classification_report(y_test, prediction))
    # print out the accuracy score
    print("accuracy:", accuracy_score(y_test, prediction))
   
