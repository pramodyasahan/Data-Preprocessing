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

