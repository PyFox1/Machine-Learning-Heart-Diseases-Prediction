#Heart Disease UCI
# The data has been obtained from https://archive.ics.uci.edu/ml/datasets/Heart+Disease

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# read csv
heartdisease = pd.read_csv("heart.csv")

# rename columns
heartdisease.columns = ['age', 'sex', 'chestpain type', 'blood pressure', 'cholestoral', 'blood sugar', 'electrocardiographic', 'max heart rate', 'angina', 'ST curve depression', 'ST slope', 'vessels number', 'thal', 'target']
a = pd.get_dummies(heartdisease['chestpain type'], prefix = "cp")
b = pd.get_dummies(heartdisease['thal'], prefix = "thal")
c = pd.get_dummies(heartdisease['ST slope'], prefix = "slope")
frames = [heartdisease, a, b, c]
df = pd.concat(frames, axis = 1).drop(columns = ['chestpain type', 'thal', 'ST slope'])

# determine x,y variables
y = df.target.values
x = df.drop(['target'], axis = 1)

# split dataset in training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# support Vector Machine
svc = SVC(random_state = 1)
svc = Pipeline([
('scaler', StandardScaler()),
('pca', PCA(n_components=4)),
('svm', SVC(random_state = 1))
])

svc.fit(x_train, y_train)
print("Test Accuracy of the SVM model: {:.2f}%".format(svc.score(x_test,y_test)*100))