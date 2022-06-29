# 3nxnFeQLJQzPbZY3
Machine learning project - Term Deposit Marketing

Data Description:

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

Attributes:

age : age of customer (numeric)

job : type of job (categorical)

marital : marital status (categorical)

education (categorical)

default: has credit in default? (binary)

balance: average yearly balance, in euros (numeric)

housing: has a housing loan? (binary)

loan: has personal loan? (binary)

contact: contact communication type (categorical)

day: last contact day of the month (numeric)

month: last contact month of year (categorical)

duration: last contact duration, in seconds (numeric)

campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

Output (desired target):

y - has the client subscribed to a term deposit? (binary)


Approach and algorithm:
- This is a classification problem with a good a mount of training data.
- There are categorical data that need to be encoded into digital number like job, marital, contact, month. Education is also categorical data that show order of education levels (ordinal).
- Applying different classification methods to see which which method is the best with the given training data.
- Performing feature selection to remove unleated or unimportant data.
- Retrain the model.
- Performing k-fold cross-validation


Attempts:
In this project I use the following model: K-Nearest Neighbor, Random Forest Classifier, Decision Tree, and Neural Network.
##############################################################
Neural network:
              precision    recall  f1-score   support

          no       0.93      1.00      0.96      7414
         yes       0.00      0.00      0.00       586

    accuracy                           0.93      8000
   macro avg       0.46      0.50      0.48      8000
weighted avg       0.86      0.93      0.89      8000

5 folds cross validation:
accuracy scores: [0.93375  0.92875  0.930625 0.92375  0.92625 ]
average accuracy: 0.929

##############################################################
K-Nearest Neighbor:
precision    recall  f1-score   support

          no       0.94      0.98      0.96      7412
         yes       0.50      0.26      0.34       588

    accuracy                           0.93      8000
   macro avg       0.72      0.62      0.65      8000
weighted avg       0.91      0.93      0.92      8000

5 folds cross validation:
accuracy scores: [0.925625 0.915625 0.926875 0.929375 0.923125]
average accuracy: 0.924

##############################################################
Decision Tree:
              precision    recall  f1-score   support

          no       0.96      0.95      0.96      7422
         yes       0.43      0.44      0.43       578

    accuracy                           0.92      8000
   macro avg       0.69      0.70      0.70      8000
weighted avg       0.92      0.92      0.92      8000

5 folds cross validation:
accuracy scores: [0.916875 0.910625 0.91125  0.913125 0.905625]
average accuracy: 0.911

##############################################################
Random Frorest Classifier:
              precision    recall  f1-score   support

          no       0.95      0.99      0.96      7438
         yes       0.56      0.25      0.34       562

    accuracy                           0.93      8000
   macro avg       0.75      0.62      0.65      8000
weighted avg       0.92      0.93      0.92      8000

5 folds cross validation:
accuracy scores: [0.929375 0.93625  0.930625 0.933125 0.925   ]
average accuracy: 0.931

##############################################################

Experiment Result:
All classification model yeild accuracy, percision, and f1-score above 91%.
The Random Forest Classifier, and Neural Network provide the best accuracy after 5-fold cross-validation.

Discussion:
The model's performance can impove by continue to remove unimportant features and provide more training data.
Random Forest Classifier out perform other classification method. It doing great with small batch of training data. 


