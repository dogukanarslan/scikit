from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

bc = load_breast_cancer()

x = scale(bc.data)
y = bc.target

# Separate train a test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = KMeans(n_clusters=5, random_state=0)

# Only pass the labels(x) and let algorithm separate them in different clusters
model.fit(x_train)

predictions = model.predict(x_test)
labels = model.labels_

print('labels', labels)
print('predictions', predictions)
print('accuracy', accuracy_score(y_test, predictions))
print('actual', y_test)
