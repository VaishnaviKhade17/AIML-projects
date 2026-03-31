# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_wine()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#BEFORE PCA   
model_original = LogisticRegression(max_iter=200)
model_original.fit(X _train, y_train)

y_pred_original = model_original.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)

# PCA (2D) 
pca_2 = PCA(n_components=2)
X_train_2 = pca_2.fit_transform(X_train)
X_test_2 = pca_2.transform(X_test)

model_2D = LogisticRegression(max_iter=200)
model_2D.fit(X_train_2, y_train)

y_pred_2 = model_2D.predict(X_test_2)
acc_2D = accuracy_score(y_test, y_pred_2)

# PCA (3D)
pca_3 = PCA(n_components=3)
X_train_3 = pca_3.fit_transform(X_train)
X_test_3 = pca_3.transform(X_test)

model_3D = LogisticRegression(max_iter=200)
model_3D.fit(X_train_3, y_train)

y_pred_3 = model_3D.predict(X_test_3)
acc_3D = accuracy_score(y_test, y_pred_3)

# PRINT RESULTS
print("Accuracy Before PCA       :", acc_original)
print("Accuracy After PCA (2D)  :", acc_2D)
print("Accuracy After PCA (3D)  :", acc_3D)

# VISUALIZATION 

# 2D Visualization
plt.figure()
plt.scatter(X_train_2[:, 0], X_train_2[:, 1], c=y_train)
plt.title("PCA 2D Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train_3[:, 0], X_train_3[:, 1], X_train_3[:, 2], c=y_train)
ax.set_title("PCA 3D Projection")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()

# PERFORMANCE COMPARISON GRAPH 
labels = ['Before PCA', 'PCA 2D', 'PCA 3D']
accuracy = [acc_original, acc_2D, acc_3D]

plt.figure()
plt.bar(labels, accuracy)
plt.title("Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Method")
plt.show()
