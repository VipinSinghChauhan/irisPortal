from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
iris = load_iris()

X = iris.data
y = iris.target

# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=42)

model = KNeighborsClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(X_train[0])

dump(model, filename="iris.joblib")
# print(model.score(X_train,y_train))
# print(model.score(X_test,y_test))
