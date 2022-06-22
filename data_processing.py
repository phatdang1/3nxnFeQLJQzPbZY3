from sys import prefix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder



# Function to read input data from csv file and make it ready for training
def readAndProcessCsv(filename):
    #loading data
    csv_data = pd.read_csv(filename)

    #list_of_data = csv_data.columns.values.tolist()
    return csv_data


# Function to encode categorical variables that is nominal data (no order)
def encodingOneHotVector(data, itemsToEncode):
    #ohe = OneHotEncoder(handle_unknown=isIgnoreUnknown,sparse=False)
    #return pd.DataFrame(ohe.fit_transform(data[[itemsToEncode]], prefix=[itemsToEncode]))
    return pd.get_dummies(data, prefix=itemsToEncode)

# Function to fill unknown or missing value with zero
def unknownToZero(data, col):
    data[col] = data[col].fillna(0)
    return data
    
# Function to prepare tesing and training data
def prepTrainingData(data, result, testPercent, classifierType):

    # setup training and testing data
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=testPercent)

    # choose a classification method
    if classifierType == "KNN":
        classifier  = KNeighborsClassifier()
    elif classifierType == "RFC":
        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
    elif classifierType == "DT":
        classifier = DecisionTreeClassifier()
    else:
        classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)

    # training the classifier
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test

# Function to drop a column and assign the column to a diffrent variable
def dropColumn(data, col):
    X = data.drop([col], axis=1)
    y = data[col]
    return X, y

