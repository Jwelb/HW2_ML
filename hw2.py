from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np 
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

# Initialize the KElbowVisualizer
model = KMeans(random_state=0)

visualizer = KElbowVisualizer(model, k=(2, 11), metric='silhouette', timings=False)

visualizer.fit(X)
visualizer.show()


best_k = visualizer.elbow_value_


best_kmeans = KMeans(n_clusters=best_k, random_state=0)
best_kmeans.fit(X)
y_pred = best_kmeans.labels_


accuracy = accuracy_score(y_true, y_pred)
confusion_mat = confusion_matrix(y_true, y_pred)

print(f'Best K: {best_k}')
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(confusion_mat)

plt.figure(figsize=(6, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

