"""This will be a script for preliminary exploration of the data"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

# We want to compare a the histogram of a variable for the passengers
# that survived verses those who didn't.
var = 'Age'
# so we have to set 'x' to the values of var which are associated with
# passengers who survived.
#x = df[var]
mask = (df['Survived'] == 1) & (df['Sex'] == 'male')
x = df.loc[mask, var]
plt.subplot(2, 1, 1)
# plt.xlim([0, 3])
# plt.ylim((0, 300))
plt.hist(x.dropna(), bins=20)
plt.xlabel('Age of male survivors')

#var = 'Parch'
#x = df[var]
mask = (df['Survived'] == 1) & (df['Sex'] == 'female')
x = df.loc[mask, var]
# y = df['Survived']
# plt.axes([0, 0, .5, .5])
plt.subplot(2, 1, 2)
# plt.xlim([0, 550])
# plt.ylim((0, 300))
plt.hist(x.dropna(), bins=20)
plt.xlabel('Age of female survivors')
plt.show()

# prints all passengers older than 65
# print(df.loc[lambda dframe: dframe.Age > 65])
