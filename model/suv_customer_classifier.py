"""
Importing libraries
"""

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading dataset in.
dataset = pd.read_csv("./dataset/Social_Network_Ads.csv")

# Initialising features and dependent variable vectors
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting up the training and testing sets using a 1/4 -> 3/4 ratio (3/4) of the dataset will be trained, (1/4) will be used for testing
# This is for an even split, even though the recommended is usually 1/5 -> testing, 4/5 -> training.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# print(f'{x_test} testing features')
# print(f'{y_test} testing dependent variable')

# print(f'{x_train} training features')
# print(f'{y_train} training dependent variable')

# Implementing feature scaling:

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print(x_train)
# print(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

# The two values in the 2D dimensional array are the values based on the layout of the dataset.
# Here, we're predicting the first observation from x_test
# Then we transform to scale it.

classifier.predict(sc.transform([[30, 87000]]))

# Predicting the values and then we're reshaping them to  display them as a singular column (Transpose)
# Prediction, reshaping and concatenation was retrieved from a previously made multiple linear regression model.

y_pred = classifier.predict(x_test)
reshaped_pred = y_pred.reshape(len(y_pred), 1)
reshaped_test = y_test.reshape(len(y_test), 1)

# print(np.concatenate((reshaped_pred, reshaped_test), 1))

# Creating the confusion matrix.

"""
A confusion matrix is a 2D matrix which shows the correct predictions and the wrong
predictions for comparisons
"""

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred)) # Retrieved a 89% accuracy score (Evaluation)
