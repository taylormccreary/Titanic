"""This will be a script for preliminary exploration of the data"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

# The following code creates a histogram of the ages of the Titanic passengers
# x = df['Age']
# y = df['Survived']
# plt.axes([0, 0, .5, .5])
# plt.hist(x.dropna(), bins=15)
# plt.xlabel('Age')
# plt.show()

# prints all passengers older than 65
# print(df.loc[lambda dframe: dframe.Age > 65])
