#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:19:32 2025

@author: ibrahim
"""

import numpy as np 
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False) 
X = mnist["data"]
y = mnist["target"].astype(np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 

# Random Forest
rf_model = RandomForestClassifier(random_state=42) 
rf_model.fit(X_train, y_train) 
rf_val_pred = rf_model.predict(X_val) 
rf_val_accuracy = accuracy_score(y_val, rf_val_pred)
print("Random Forest Validation Accuracy:", rf_val_accuracy)

# SVM 
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train) 
svm_val_pred = svm_model.predict(X_val) 
svm_val_accuracy = accuracy_score(y_val, svm_val_pred)
print("Support Vector Machine Validation Accuracy:", svm_val_accuracy)

# Choose the best model based on validation accuracy.
if rf_val_accuracy > svm_val_accuracy:
    best_model = rf_model
    print("We are going to use Random Forest in our application.")
else:
    best_model = svm_model
    print("We are going to use Support Vector Machine in our application.")

# Save the best model
import joblib
joblib.dump(best_model, 'mnist_model.pkl')

# Test the best model in data test
test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print(f"Test Accuracy with the best model: {test_accuracy}")