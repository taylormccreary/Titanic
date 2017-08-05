"""test"""
import pandas as pd


def read_training_data(training_csv):
    """
    Input: csv file with all the training data
    Output: a numpy array(?) containing the training data
    """
    train_df = pd.read_csv(training_csv)
    return train_df

TRAIN_DATA = read_training_data('train.csv')
