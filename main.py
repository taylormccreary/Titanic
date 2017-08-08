"""This is the main script which will be executed"""
import pandas as pd

def get_survival(passenger, gender_probs):
    p_gender = passenger['Sex']
    return int(round(gender_probs.loc[p_gender,'Survived']))

# read in training data
train_df = pd.read_csv('train.csv')

"""This is the part where I need to build a model"""

# read in test data
test_df = pd.read_csv('test.csv')

# gender_df will be my model used to predict survival based solely on gender
gender_df = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()
print(gender_df.loc['female','Survived'])
print(gender_df.loc['male','Survived'])


"""This is the part where I need to use my model to predict survival on the test data"""

# output_df is the pandas DataFrame that contains the final output
output_df = pd.DataFrame(index=test_df['PassengerId'])

# currently, just predicting that everyone dies
output_df['Survived'] = 0 # this must be a float

# iterate through the test data, using the model to compute the Survival
for index, row in test_df.iterrows() :
    # get the output row corresponding to this row in test_df
    # that row, the 'Survived' column should be set to be a function of the test_df row
    testval = get_survival(row, gender_df)
    #print(testval)
    ind = row['PassengerId']
    output_df.set_value(ind, 'Survived', testval)

# exporting the DataFrame of ids and predictions into a .csv
output_df.to_csv('submission.csv')