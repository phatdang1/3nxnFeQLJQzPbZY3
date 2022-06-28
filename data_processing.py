from sys import prefix
import pandas as pd



#from sklearn.preprocessing import OneHotEncoder



# Function to read input data from csv file and make it ready for training
def readAndProcessCsv(filename, to_list):
    #loading data
    csv_data = pd.read_csv(filename)

    if(to_list):
        return csv_data.columns.values.tolist()
    else:
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
    
# Function to drop a column and assign the column to a diffrent variable
def dropColumn(data, col):
    X = data.drop([col], axis=1)
    y = data[col]
    return X, y

