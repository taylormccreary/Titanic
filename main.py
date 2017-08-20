"""This is the main script which will be executed"""
import pandas as pd
from sklearn import tree

# read in training data
train_df = pd.read_csv('train.csv')

# Now let's make a decision tree with it!
clf = tree.DecisionTreeClassifier()

# in order to do this, we need to clean up the data first

# make all the male and female values for Sex 1s and 0s
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0})
train_df['Sex'].fillna(-1, inplace=True)
# Make all the NaN Pclass values -1
train_df['Pclass'].fillna(-1, inplace=True)
# Make all the NaN Age values -1
train_df['Age'].fillna(-1, inplace=True)
get_agegrp = lambda x: 0 if x < 5 and x >=0 else 1
train_df['AgeGrp'] = train_df['Age'].map(get_agegrp)
# Make all the NaN Fare values -1
train_df['Fare'].fillna(-50, inplace=True)
get_faregrp = lambda x: round(x / 50.0)
train_df['FareGrp'] = train_df['Fare'].map(get_faregrp)

#clf = clf.fit(train_df.loc[:, ['Pclass', 'Sex']], train_df.loc[:, 'Survived'])
clf = clf.fit(train_df[['AgeGrp']], train_df['Survived'])

def get_survival(passenger):
    """ Takes as input a row of a data frame corresponding to a passenger,
    and uses a decision tree classifier to assign a value to 'Survived' """
    passenger = passenger[['AgeGrp']].values.reshape(1, -1)
    res = clf.predict(passenger)
    return int(res[0])

# read in test data
test_df = pd.read_csv('test.csv')
# clean test data as well
# make all the male and female values for Sex 1s and 0s
test_df['Sex'].replace(['female', 'male'], [1, 0], inplace=True)
test_df['Sex'].fillna(-1, inplace=True)
# Make all the NaN Pclass values -1
test_df['Pclass'].fillna(-1, inplace=True)
# Make all the NaN Age values -1
test_df['Age'].fillna(-1, inplace=True)
test_df['AgeGrp'] = test_df['Age'].map(get_agegrp)
# Make all the NaN Fare values -1
test_df['Fare'].fillna(-50, inplace=True)
test_df['FareGrp'] = test_df['Fare'].map(get_faregrp)

# iterate through the test data, using the model to compute the Survival
# the Survived column is being added to the test data DataFrame
test_df['Survived'] = test_df.apply(get_survival, axis=1)

# exporting the DataFrame of ids and predictions into a .csv
test_df[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
