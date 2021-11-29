# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                data, digits.target, test_size= 0.2, shuffle=False
            )

X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,
    y_test_valid, test_size=0.5, shuffle=False,
)

trainsize = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
acclst = []
for i in trainsize:
    X_train1, X_test_valid1, y_train1, y_test_valid1 = train_test_split(
                X_train, y_train, test_size= i, shuffle=False
            )
    clf.fit(X_train1, y_train1)
    predicted_valid = clf.predict(X_test)
    f1_valid = metrics.f1_score(
                y_pred=predicted_valid, y_true=y_test, average="macro"
            )
    acclst.append(f1_valid)

ts = []
for i in trainsize:
    ts.append(1-i)

print(acclst)
plt.plot(ts,acclst)
plt.title('model accuracy')
plt.xlabel('training split size')
plt.ylabel('accuracy')
plt.show()

# Step 5-  observation from the chart is as we increase the size of training set, accuracy of the model

print("Comparison between 10 and 30 percnt training data")
trainsize = [0.7,0.9]
acclst = []
for i in trainsize:
    X_train1, X_test_valid1, y_train1, y_test_valid1 = train_test_split(
                X_train, y_train, test_size= i, shuffle=False
            )
    clf.fit(X_train1, y_train1)
    predicted_valid = clf.predict(X_test)
    print(f'Confusion matrix of {1-i} training data')
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

# =======================================================================

print("Comparison between 10 and 20 percnt training data")
trainsize = [0.8,0.9]
acclst = []
for i in trainsize:
    X_train1, X_test_valid1, y_train1, y_test_valid1 = train_test_split(
                X_train, y_train, test_size= i, shuffle=False
            )
    clf.fit(X_train1, y_train1)
    predicted_valid = clf.predict(X_test)
    print(f'Confusion matrix of {1-i} training data')
    disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")