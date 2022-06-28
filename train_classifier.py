from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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