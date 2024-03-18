import pandas as pd
import numpy as np
import random

def ingest_sputtering_data(fpath,train_variables = None, predict_variables = None, arr_type = 'numpy'):

    # takes a filepath and returns a numpy array or pandas dataframe with the data
    # first open the file

    # train_variables and predict_variables are lists of strings with the column headers
    # that you want to use as an input/output variable
    
    df = pd.read_csv(fpath,header=0)

    # if they don't specify any variables then just exit the funciton
    if train_variables == None or predict_variables == None:
        return

    # if they just want the pandas array then return that
    if arr_type == 'pd' or arr_type == 'pandas':
        return df
    
    elif arr_type == 'np' or arr_type == 'numpy':
        X_all = np.array([list(df[var]) for var in train_variables])
        Y_all = np.array([list(df[var]) for var in predict_variables])

        return X_all,Y_all
    

# a function for scaling input variables
# each input variable will be scaled between 1 and zero based on the highest observed value for that variable
def scale_input_data(X_data,scale_values = None):
    # X_data is a numpy array where each row is an input vector and each column is a list
    # of sputtering gun settings (power, angle, etc)

    # if scale_values are provided then this will scale the values of X_data by the entries in scale_values
    # if scale_values is None then this will scale the values of X_data by the maximal element in each column of X_data
    # scale_values == None will also return an array of the maximal elements

    if scale_values == None:

        # find the maximal element in each column
        X_scale = [np.max(X) for X in X_data]

        # now scale the values
        X_new = np.array([X/scale for X,scale in zip(X_data,X_scale)])
        X_new = np.nan_to_num(X_new)

        return X_new,X_scale
    
    else:
        X_new = np.array([X/scale for X,scale in zip(X_data,scale_values)])
        return X_new

# function for generating a random test-train split from an input vector X and output vector Y
def test_train_split(X_complete,Y_complete,split = 10,return_rand = False):

    # split is the number of samples to use in the training data
    # the remainder will be the test data

    # take transposes
    X_complete = X_complete.T
    Y_complete = Y_complete.T

    # get the shape of the arrays
    X_shape = X_complete.shape
    Y_shape = Y_complete.shape

    # generate an array of random variables in the range
    rand_vals = random.sample(range(X_shape[0]), X_shape[0])

    # split the random variables into two sets
    train_vals = np.array(rand_vals[0:split])
    test_vals = np.array(rand_vals[split:X_shape[0]])

    # now pull out the associated data points into separate X and Y arrays
    X_train = X_complete[train_vals,:]
    Y_train = Y_complete[train_vals,:]

    X_pred = X_complete[test_vals,:]
    Y_pred = Y_complete[test_vals,:]

    if return_rand == True:
        return X_train, Y_train, X_pred, Y_pred, rand_vals
    else:
        return X_train, Y_train, X_pred, Y_pred
    

    import itertools

# this is a function written by gpt4 that finds all subsets of a dataset that have non zero values
# it will find all combinations of elements that have non zero input values 
def non_zero_combinations(dataset):
    result = {}
    subset = {}
    
    # Iterate through all possible index combinations from size 1 to 6
    for size in range(1, 7):
        for indices in itertools.combinations(range(6), size):
            # Collect non-zero entries for the current index combination
            non_zero_values = []
            non_zero_subset = []
            for j,vector in enumerate(dataset):
                if all(vector[i] != 0 for i in indices):
                    non_zero_values.append(vector)
                    non_zero_subset.append(j)
            result[indices] = non_zero_values
            subset[indices] = non_zero_subset

    return result,subset