import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,tree
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import DataConversionWarning
from skimage.transform import rescale, resize, downscale_local_mean
import pandas as pd

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#5 diff split 
gamma_p = [1e-7,1e-5,1e-3,0.01,0.1,1]
split = [0.15]
#list
acc_svm=[]
bestparam_svm=[]

temp_acc =[]



def acc_cal_svm(gamma_p):

    clf = svm.SVC(gamma=gamma_p)
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(
                    data, digits.target, test_size=0.15 + 0.15, shuffle=False
                )

    X_test, X_valid, y_test, y_valid = train_test_split(
    X_test_valid, y_test_valid, test_size=0.15 / (0.15 + 0.15),shuffle=False,
    )
    clf.fit(X_train,y_train)
    predicted_valid = clf.predict(X_valid)
    acc_valid = metrics.accuracy_score(y_pred=predicted_valid, y_true=y_valid)
    f1_valid = metrics.f1_score(
        y_pred=predicted_valid, y_true=y_valid, average="macro"
    )

    # we will ensure to throw away some of the models that yield random-like performance.
    # if acc_valid < 0.11:
    #     print("Skipping for {}".format(gamma_p))

    return acc_valid

badparam_svm = []
badacc_svm = []

for i in range(3):   
    temp_acc=[]
    for j in gamma_p:
        acc=acc_cal_svm(j)
        temp_acc.append(acc)
    print(temp_acc)
    max_acc_index=temp_acc.index(max(temp_acc))
    max_acc=max(temp_acc)
    best_gamma=gamma_p[max_acc_index]
    bestparam_svm.append(best_gamma)
    acc_svm.append(max_acc)

    min_acc_index = temp_acc.index(min(temp_acc))
    min_acc = min(temp_acc)
    bad_gamma = gamma_p[min_acc_index]
    badparam_svm.append(bad_gamma)
    badacc_svm.append(min_acc)

# # print(split)
noOftimes = [1,2,3]
# print(bestparam_svm)
# print(acc_svm)

data = {"round":noOftimes, "optimal_gamma": bestparam_svm, "best_acc_svm" : acc_svm, "bad_gamma":badparam_svm, "bad_acc_svm": badacc_svm}

df = pd.DataFrame(data)

mean_svm=str(df["best_acc_svm"].mean())
mean_badsvm = str(df["bad_acc_svm"].mean())

std_svm = str(round(df["best_acc_svm"].std(),4))
std_badsvm = str(round(df["bad_acc_svm"].std(),4))

plusminus_symbol="\u00B1"

              
ms = mean_svm+plusminus_symbol+std_svm
md = mean_badsvm + plusminus_symbol + std_badsvm

s = pd.Series([' ',' ', ms, '',md], index = ['round', 'optimal_gamma',' best_acc_svm', 'bad_gamma', 'bad_acc_svm'])
df = df.append(s, ignore_index = True)
print(df)
