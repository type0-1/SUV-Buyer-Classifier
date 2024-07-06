"""
Importing libraries
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split

# Loading dataset in.
dataset = pd.read_csv("./dataset/Social_Network_Ads.csv")

# Initialising features and dependent variable vectors
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting up the training and testing sets using a 1/4 -> 3/4 ratio (3/4) of the dataset will be trained, (1/4) will be used for testing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
