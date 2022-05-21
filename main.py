from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# Split it in features and labels
X = iris.data
y = iris.target

# Hours of study vs good/bad grades
# Consider 10 different students, train with 8 and predict with 2 students

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
