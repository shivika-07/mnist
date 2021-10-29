import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning
import pandas as pd


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


gammaList = [1e-7,1e-5,1e-3,0.01,0.1,1]
depth = [5,10,25,50,100]
testSize = [0.10, 0.20, 0.35, 0.70, 0.9]


def calSVMAcc(gamma_p, testSize):
  clf = svm.SVC(gamma=gamma_p)
  X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=testSize, shuffle=False)
  X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=testSize, shuffle=False)
  clf.fit(X_train,y_train)
  acc = clf.score(X_val,y_val)
  return acc

def calTreeAcc(depth, testSize):
  treeclassifier = tree.DecisionTreeClassifier(max_depth = depth)
  X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=testSize, shuffle=False)
    
  X_train, X_val, y_train, y_val = train_test_split(
        data, digits.target, test_size=testSize, shuffle=True)
  
  treeclassifier.fit(X_train,y_train)
  predicted = treeclassifier.predict(X_val)
  acc = metrics.accuracy_score(y_val, predicted)
  return acc

acc_svm=[]
bestparam_svm=[]
acc_tree=[]
bestparam_tree=[]

# computing svm accuracy
for ts in testSize:
  temp_acc=[]
  for gammaval in gammaList:
    acc= calSVMAcc(gammaval,ts)
    temp_acc.append(acc)

  accVal = max(temp_acc)
  max_acc_index= temp_acc.index(accVal)
  bestparam_svm.append(gammaList[max_acc_index])
  acc_svm.append(max(temp_acc))

# computing decision tree accuracy
for ts in testSize:
  temp_acc=[]
  for dep in depth:
    acc= calTreeAcc(dep,ts)
    temp_acc.append(acc)

  accVal = max(temp_acc)
  max_acc_index=temp_acc.index(accVal)
  bestparam_tree.append(depth[max_acc_index])
  acc_tree.append(max(temp_acc))


data = {"testSize":testSize, "bestGammaVal": bestparam_svm, "Acc_SVM" : acc_svm, "bestDepth": bestparam_tree, "Acc_DecisionTree": acc_tree}
df = pd.DataFrame(data)

SVMmean=str(df["Acc_SVM"].mean())
TreeMean=str(df["Acc_DecisionTree"].mean())
SVMStd = str(round(df["Acc_SVM"].std(),2))
TreeStd = str(round(df["Acc_DecisionTree"].std(),2))
             
s = pd.Series([' ',' ', SVMmean+'\u00B1'+SVMStd, ' ', TreeMean+'\u00B1'+TreeStd], index = ['testSize', 'bestGammaVal','Acc_SVM', 'bestDepth', 'Acc_DecisionTree'])
df = df.append(s,ignore_index = True)
print(df)