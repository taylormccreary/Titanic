"""This will be a script for preliminary exploration of the data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# x = np.linspace(0, 1, 201)
# y = np.sin((2*np.pi*x)**2)

# plt.plot(x, y, 'red')
# plt.show()



df = pd.read_csv('train.csv')
x = df['Age']
y = df['Survived']
#plt.scatter(x, y)
plt.hist(x.dropna(), bins=15)
plt.show()

#for index, row in df.iterrows() :
#    print(row['Sex'], row['Survived'])

#print(df.columns.values)
#print(df.describe())

#df.plot()
#plt.show()
