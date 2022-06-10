from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create dataset
X, y = make_blobs(n_samples=150, n_features=2, centers=3,
                  cluster_std=0.5, shuffle=True, random_state=0)

# Plot blobs
plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50, )
plt.show()

km = KMeans(n_clusters=3, init='random', n_init=10,
            max_iter=300, tol=1e-04, random_state=0)

# Predict each points label among 0, 1 and 2
y_km = km.fit_predict(X)

# Plot points which belong to 0 labeled cluster
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen',
            marker='o', edgecolor='black', label='cluster 1')

# Plot points which belong to 1 labeled cluster
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange',
            marker='v', edgecolor='black', label='cluster 2')

# Plot points which belong to 2 labeled cluster
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue',
            marker='*', edgecolor='black', label='cluster 3')

# Plot centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[
            :, 1], s=250, marker='*', c='red', edgecolor='black', label='centroids')

plt.legend(scatterpoints=1)
plt.grid()
plt.show()
