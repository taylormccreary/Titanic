"""This will be a script for preliminary exploration of the data"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')


#for index, row in df.iterrows() :
#    print(row['Sex'], row['Survived'])

print(df.columns.values)
print(df.describe())

#df.plot()
#plt.show()
