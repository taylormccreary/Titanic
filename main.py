"""This is the main script which will be executed"""
import pandas as pd

# read in training data
train_df = pd.read_csv('train.csv')
print(list(train_df.columns.values))

"""This is the part where I need to build a model"""

# read in test data
test_df = pd.read_csv('test.csv')
print(list(test_df.columns.values))

"""This is the part where I need to use my model to predict survival on the test data"""

# output_df is the pandas DataFrame that contains the final output
output_df = pd.DataFrame(index=test_df['PassengerId'])

# currently, just predicting that everyone dies
output_df['Survived'] = 0

# exporting the DataFrame of ids and predictions into a .csv
output_df.to_csv('submission.csv')