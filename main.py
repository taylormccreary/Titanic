"""This is the main script which will be executed"""
import pandas as pd


global train_df
train_df = pd.read_csv('train.csv')
print(list(train_df.columns.values))

#print(train_df.info())

SURVIVED_CT = train_df['Survived']
#print(train_df['Cabin'])

test_df = pd.read_csv('test.csv')
print(list(test_df.columns.values))

output_df = pd.DataFrame(index=test_df['PassengerId'])
output_df['Survived'] = 0
#output_df.index = output_df['PassengerId']
print(list(output_df.columns.values))
print(output_df)
output_df.to_csv('submission.csv')
