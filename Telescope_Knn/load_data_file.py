import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import classification_telescopedata
import classification_telescopedata_sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

def load_data():
    data = pd.read_csv(r'C:\Users\me\Desktop\Engineering_Term _7\Machine_Learning\Assignment1\Telescope_Knn\telescope_data\telescope_data.csv', header=0)

    # Minimize gamma class randomly
    gamma_rows = data[data['class']== 'g']
    hadron_rows = data[data['class']== 'h']

    g_minimized = gamma_rows.sample(n=6688, random_state = 40 )

    # Combine the minimized gamma class with the hadron class, samples randomly and resets indices
    dataset = pd.concat([g_minimized, hadron_rows], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    X= dataset.drop(columns=['class']).iloc[:, 1:].values
    y= dataset['class'].values

    # First split: 85% train+validation sets, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=40)

    # Second split: from the 85%, take 15/85 ≈ 0.176 for validation
    X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=0.176, random_state=40)

    #Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_validation = sc_X.transform(X_validation)
    X_test = sc_X.transform(X_test)
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test
