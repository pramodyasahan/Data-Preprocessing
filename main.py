import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :3].values
y = dataset.iloc[:, -1].values

impute = SimpleImputer(missing_values=np.nan, strategy="mean")
impute.fit(X[:, 1:3])
X[:, 1:3] = impute.transform(X[:, 1:3])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

le = LabelEncoder()
y = le.fit_transform(y)

