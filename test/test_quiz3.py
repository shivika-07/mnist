from sklearn import datasets
import numpy as np
from joblib import dump, load

def  test_digit_correct_0svm():
	digits = datasets.load_digits()
	data=digits.images; target=digits.target
	data = data.reshape((digits.images.shape[0], -1))
	res=np.where(target==0); op=res[0]
	clf=load("model_svm.joblib")
	pred=clf.predict(data[op])
	assert pred[0]==0

def  test_digit_correct_1svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==1); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==1

def  test_digit_correct_2svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==2); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==2
        
def  test_digit_correct_3svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==3); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==3

def  test_digit_correct_4svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==4);op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==4

def  test_digit_correct_5svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==5); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[1]==5

def  test_digit_correct_6svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==6); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==6

def  test_digit_correct_7svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==7); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==7
def  test_digit_correct_8svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==8); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==8
def  test_digit_correct_9svm():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==9); op=res[0]
        clf=load("model_svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==9



def  test_digit_correct_0dt():
	digits = datasets.load_digits()
	data=digits.images; target=digits.target
	data = data.reshape((data.shape[0], -1))
	res=np.where(target==0); op=res[0]
	clf=load("model_decisiontree.joblib")
	pred=clf.predict(data[op])
	assert pred[0]==0

def  test_digit_correct_1dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==1); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==1
def  test_digit_correct_2dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==2);  op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==2
def  test_digit_correct_3dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==3); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==3
def  test_digit_correct_4dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==4); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==4
def  test_digit_correct_5dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==5); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[1]==5
def  test_digit_correct_6dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==6);  op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==6

def  test_digit_correct_7dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==7);op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==7
def  test_digit_correct_8dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==8); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==8
def  test_digit_correct_9dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==9); op=res[0]
        clf=load("model_decisiontree.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==9
