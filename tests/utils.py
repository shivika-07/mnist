from sklearn.model_selection import train_test_split
from skimage.transform import rescale
from sklearn import datasets, metrics, svm
import numpy as np
import os
from joblib import dump, load

def preprocess(images, rf):
    resized_images = []
    updated_images = []
    for d in images:
        resized_images.append(rescale(d, rf, anti_aliasing=False))
    updated_images = np.asarray(resized_images)
    updated_images = updated_images.reshape((len(images), -1))
    return updated_images

def create_split(data,target, test_size,valid_size):
    X_train, X_val, y_train, y_val = train_test_split(
            data, target, test_size = test_size + valid_size, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size= valid_size / (test_size + valid_size), shuffle=False)

    return X_train, X_test, X_val, y_train, y_test, y_val

def verify(clf, X, y):
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred = predicted, y_true=y)
    f1 = metrics.f1_score(y_pred = predicted, y_true=y, average="macro")
    return {'acc':acc , 'f1':f1}

def get_random_acc(y):
    return max(np.bincount(y))/len(y)

def run_classification_experiment(X_train,y_train,X_val,y_val):
    clf = svm.SVC(gamma=0.01, max_iter=1000)
    clf.fit(X_train,y_train)
    train_metric = verify(clf, X_train, y_train)
    val_metric = verify(clf, X_val, y_val)
    return train_metric['acc'],train_metric['f1'] ,val_metric['acc'], val_metric['f1']

def model_creation(X_train,y_train):
    gamma=0.01
    clf=svm.SVC(gamma=gamma, max_iter=1000)
    clf.fit(X_train,y_train)
    output_folder = "./models/gamma_{}".format(gamma)
    try:
        os.mkdir(output_folder)
    except:
        pass
    dump(clf, os.path.join(output_folder,"model.joblib"))
    return output_folder