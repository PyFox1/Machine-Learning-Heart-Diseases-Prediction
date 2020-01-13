#Heart Disease UCI
# The data has been obtained from https://archive.ics.uci.edu/ml/datasets/Heart+Disease

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

heartdisease = pd.read_csv("heart.csv")
heartdisease.head()

# How many tupels and how many attributes do we have?
print(heartdisease.shape)
print(heartdisease.size)

# Set meaningful column names
heartdisease.columns = ['age', 'sex', 'chestpain type', 'blood pressure', 'cholestoral', 'blood sugar', 'electrocardiographic', 'max heart rate', 'angina', 'ST curve depression', 'ST slope', 'vessels number', 'thal', 'target']

# Look at the top lines of the data table
print(heartdisease.head())
print(heartdisease.describe())
heartdisease["blood pressure"].describe()

# Seaborn Plot
sns.countplot(x="target", data=heartdisease);

# split data in test and training data
x = df.drop(['target'], axis = 1)
y = df.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

# Descision Trees
dtc = DecisionTreeClassifier(max_depth=8)
dtc.fit(x_train, y_train)
print("Decision Tree Model Accuracy {:.2f}%".format(dtc.score(x_test, y_test)*100))

plt.figure(figsize=(20,10))
tree.plot_tree(dtc.fit(x_train, y_train), max_depth=6,fontsize=11, filled=True)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(dtc, x, y, cv=5, scoring='recall_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))