"""This is the main script which will be executed"""
import pandas as pd


# read in training data
train_df = pd.read_csv('train.csv')

"""This is the part where I need to build a model"""

# gender_df will be my model used to predict survival based solely on gender
# get_survival will be the function that uses a gender based model to predict survival
gender_df = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()

def get_survival(passenger):
    """ Takes as input a row of a data frame corresponding to a passenger,
    and a data frame with the probabilites of survival by gender"""
    p_gender = passenger['Sex']
    res = int(round(gender_df.loc[p_gender, 'Survived']))
    return res


#"""This is the part where I need to use my model to predict survival on the test data"""

# read in test data
test_df = pd.read_csv('test.csv')

# iterate through the test data, using the model to compute the Survival
# the Survived column is being added to the test data DataFrame
# fix these red squiggles!
test_df['Survived'] = test_df.apply(get_survival, axis=1)

# exporting the DataFrame of ids and predictions into a .csv
test_df[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)