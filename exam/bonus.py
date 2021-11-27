import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# case1 - 
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
acclst = []
for gamma in [10 ** exponent for exponent in range(-7, 0)]:

    # clf = MLPClassifier(gamma)

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    data, digits.target, test_size= 0.2, shuffle=False
                )

    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,
        y_test_valid, test_size=0.5, shuffle=False,
    )

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    predicted_valid = clf.predict(X_test)
    f1_valid = metrics.f1_score(
                y_pred=predicted_valid, y_true=y_test, average="macro"
            )
    acclst.append(f1_valid)

print("case1-accuracy")
print(acclst)


# case 2 -
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
size = [0.1,0.4,0.3,0.2]
for gamma in [10 ** exponent for exponent in range(-7, 0)]:

    # clf = MLPClassifier(gamma)

    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    data, digits.target, test_size= size, shuffle=False
                )

    X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,
        y_test_valid, test_size=0.5, shuffle=False,
    )

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    predicted_valid = clf.predict(X_test)
    f1_valid = metrics.f1_score(
                y_pred=predicted_valid, y_true=y_test, average="macro"
            )
    acclst.append(f1_valid)

print("Case2- accuracy")
print(acclst)
