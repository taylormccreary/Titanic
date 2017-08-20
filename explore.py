"""This will be a script for preliminary exploration of the data"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

# We want to compare a the histogram of a variable for the passengers
# that survived verses those who didn't.
var = 'Fare'
# so we have to set 'x' to the values of var which are associated with
# passengers who survived.
#x = df[var]
x = df.loc[(df['Survived'] == 1), var]
plt.subplot(2, 1, 1)
plt.hist(x.dropna(), bins=20)
plt.xlabel(var)

#var = 'Parch'
#x = df[var]
x = df.loc[(df['Survived'] == 0), var]
# y = df['Survived']
# plt.axes([0, 0, .5, .5])
plt.subplot(2, 1, 2)
plt.hist(x.dropna(), bins=20)
plt.xlabel(var)
plt.show()

# prints all passengers older than 65
# print(df.loc[lambda dframe: dframe.Age > 65])
