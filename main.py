"""This is the main script which will be executed"""
import pandas as pd


train_df = pd.read_csv('train.csv')
print(list(train_df.columns.values))


test_df = pd.read_csv('test.csv')
print(list(test_df.columns.values))

output_df = pd.DataFrame(index=test_df['PassengerId'])
output_df['Survived'] = 0
output_df.to_csv('submission.csv')