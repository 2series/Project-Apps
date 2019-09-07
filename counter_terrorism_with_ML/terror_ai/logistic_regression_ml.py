# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Function to split the dataset
def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.iloc[:, 0:-1]
    Y = balance_data.iloc[:, -1]
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
    return X_train, X_test, y_train, y_test


# Function to perform training with OVR
def train_using_ovr(X_train, y_train):
    # create the classifier object
    clf = LogisticRegression(random_state = 1997)
    clf.fit(X_train, y_train)
    return clf


# Function to make predictions
def prediction(X_test, clf):
    # Predicton on test with giniIndex
    y_pred = clf.predict(X_test)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    return accuracy_score(y_test,y_pred)*100


# Driver code
def logi_main(df, test_df):
    X_train, X_test, y_train, y_test = splitdataset(df)
    # train a logistic regression using one vs rest method
    clf = train_using_ovr(X_train, y_train)
    # prediction
    y_pred = prediction(X_test, clf)
    accuracy = cal_accuracy(y_test, y_pred)
    y_pred = prediction(test_df, clf)
    return y_pred, accuracy