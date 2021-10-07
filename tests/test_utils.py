import sklearn
from sklearn import datasets, svm, metrics
from utils import *

digits = datasets.load_digits()
X = digits.images
Y = digits.target
data= preprocess(digits.images, 0.5)

X_train, X_test, X_val, y_train, y_test, y_val = create_split(data , digits.target, 0.15, 0.15)

def test_model_writing():
    path = model_creation(X_train,y_train)
    assert os.path.isfile(path +"/model.joblib")

def test_small_data_overfit_checking():
    train_acc, train_f1, val_acc, val_f1 = run_classification_experiment(X_train,y_train,X_val,y_val)

    print(f'Training Accuracy is {train_acc} and f1 score is {train_f1}')
    print(f'Val Accuracy is {val_acc} and f1 score is {val_f1}')

    assert train_acc > 0.90
    assert train_f1 > 0.90
