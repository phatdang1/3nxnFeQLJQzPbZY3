from data_processing import*
from result_report import*

# model code: KNN = K-Nearest-Neighbor; DT = Decision Tree; RFC = Random Forest Classifier; NN = Neural Network
model = 'NN'
##### processing data ####

# read in data
list_of_data = readAndProcessCsv('term-deposit-marketing-2020.csv')

# encoding categorical data (data that not a number)
job = encodingOneHotVector(list_of_data.job, "job")
marital = encodingOneHotVector(list_of_data.marital, "marital")
defaults = encodingOneHotVector(list_of_data.default, "default")
housing = encodingOneHotVector(list_of_data.housing, "housing")
loan = encodingOneHotVector(list_of_data.loan, "loan")
contact = encodingOneHotVector(list_of_data.contact, "contact")
month = encodingOneHotVector(list_of_data.month, "month")

### encoding ordinal data to show the education level###

#create education dictionary with rank number
education_dict = {'primary': 1, 'secondary': 2, 'tertiary': 3}

# mapping the rank with each education level, ignore the first column
education = pd.DataFrame(list_of_data.education.map(education_dict), columns=['education'])

# fill unknown value with zero
education = unknownToZero(education, "education")

# drop categorical columns
processed_data = list_of_data.drop(['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'education'], axis=1)

# add the processed columns back
processed_data = pd.concat([processed_data, job, marital, defaults, housing, loan, contact, month, education], axis=1)



#### Train the classifier and make prediction ####

# seperate the result from label data
X, y = dropColumn(processed_data, 'y')
reportAndScore(X, y, model, 0.2)

print('5 folds cross validation:')
crossValidationNKFold(X, y, model, 0.2, 5)
