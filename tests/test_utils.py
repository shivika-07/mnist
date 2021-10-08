import sklearn
from sklearn import datasets, svm, metrics
from utils import *

digits = datasets.load_digits()
X = digits.images
Y = digits.target
data= preprocess(digits.images, 0.5)

X_train, X_test, X_val, y_train, y_test, y_val = create_split(data , digits.target, 0.15, 0.15)

# def test_model_writing():
#     path = model_creation(X_train,y_train)
#     assert os.path.isfile(path +"/model.joblib")

# def test_small_data_overfit_checking():
#     train_acc, train_f1, val_acc, val_f1 = run_classification_experiment(X_train,y_train,X_val,y_val)

#     print(f'Training Accuracy is {train_acc} and f1 score is {train_f1}')
#     print(f'Val Accuracy is {val_acc} and f1 score is {val_f1}')

#     assert train_acc > 0.90
#     assert train_f1 > 0.90


def test_sample2():
    X_train, X_test, X_val, y_train, y_test, y_val = create_split(data[:9] , digits.target[:9], 0.2, 0.1)
    assert X_train.shape[0] == 6
    assert X_test.shape[0] == 2
    assert X_val.shape[0] == 1
    assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0] == 9

def test_sample1():
    X_train, X_test, X_val, y_train, y_test, y_val = create_split(data[0:100] , digits.target[0:100], 0.2, 0.1)
    assert X_train.shape[0] == 69
    assert X_test.shape[0] == 21
    assert X_val.shape[0] == 10
    assert X_train.shape[0] + X_test.shape[0] + X_val.shape[0] == 100



