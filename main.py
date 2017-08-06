"""This is the main script which will be executed"""
import pandas as pd

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
output_df['Survived'] = 0

## NEXT: iterate & apply model
for index, row in df.iterrows() :
    print(row['A'], row['B'])

# exporting the DataFrame of ids and predictions into a .csv
output_df.to_csv('submission.csv')