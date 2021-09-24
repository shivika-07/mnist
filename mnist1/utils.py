from sklearn.model_selection import train_test_split
from skimage.transform import rescale
from sklearn import datasets, metrics, svm

def preprocess(images, rf):
    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rf, anti_aliasing=False))
    return resized_images

def create_split(data,target, test_size,valid_size):
    X_train, X_val, y_train, y_val = train_test_split(
            data, target, test_size = test_size + valid_size, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size= valid_size / (test_size + valid_size), shuffle=False)

    return X_train, X_test, X_val, y_train, y_test, y_val

def test(clf,X, y):
    predicted = clf.predict(X)

    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y, average="macro")

    return {'acc':acc , 'f1':f1}