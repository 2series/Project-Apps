# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.iloc[:, 0:-1]
    Y = balance_data.iloc[:, -1]
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 2107)
    return X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 2107)
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def train_using_entropy(X_train, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 2107)
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 

# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    return accuracy_score(y_test,y_pred)*100


# Driver code
def decision_main(df, test_df):
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = splitdataset(df)
    #clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    accuracy = cal_accuracy(y_test, y_pred_entropy)
    y_pred = prediction(test_df, clf_entropy)
    return y_pred, accuracy