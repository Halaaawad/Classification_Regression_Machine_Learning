import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

# Function to find the best k value for KNN using sklearn and the validation set
def validation_accuracy_sklearn(X_train, y_train, X_validation, y_validation):
    best_k=1
    best_acc=0
    #used for plotting later
    k_values = []
    accuracies = []
    for k in range(1, 95, 2):
        #as k must be odd
        if k%2 !=0:
            print(f"Testing k={k}")
            classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
            classifier.fit(X_train, y_train)
            y_valpredicted = classifier.predict(X_validation)
            accuracy = accuracy_score(y_validation, y_valpredicted)

            k_values.append(k)
            accuracies.append(accuracy)

            if accuracy> best_acc:
                best_acc = accuracy
                best_k = k
                print(f"New best k: {best_k}, New best accuracy for validation set: {best_acc}")
                
    print("Finished testing SKlearn KNN.\n")
    print(f"Best k (Sklearn): {best_k}, Best accuracy for validation set: {best_acc}\n")
    return best_k, best_acc, k_values, accuracies





