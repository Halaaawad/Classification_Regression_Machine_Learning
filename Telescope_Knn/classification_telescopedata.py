import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split

# Function to compute Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Manual implementation of KNN Classifier
class KNN:
    def __init__(self,k):
        self.k = k
    
    def fit (self,X,y):
        self.X_train = X
        self.y_train = y
    
    # Predict the class labels for the provided data points
    def predict(self,new_points):  
        predictions = [self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)
    
    # Helper function to predict the class for a single data point
    def predict_class(self,new_point):
        distances = [euclidean_distance(point, new_point) for point in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

# Function to find the best k value for KNN using manual implementation and the validation set  
def validation_accuracy_manual(X_train, y_train, X_validation, y_validation):
    best_k=1
    best_acc=0
    #used for plotting later
    k_values = []
    accuracies = []
    for k in range(1, 20, 2):
        #as k must be odd
        if k%2 !=0:
            print(f"Testing k={k}")
            model = KNN(k=k)
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)
            acc = np.sum(predictions == y_validation) / len(y_validation)

            k_values.append(k)
            accuracies.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_k = k
                print(f"New best k: {best_k}, New best accuracy for validation set: {best_acc}")

    print("Finished testing manual KNN.\n")
    print(f"Best k (Manual): {best_k}, Best accuracy for validation set: {best_acc}\n")
    return best_k, best_acc, k_values, accuracies





