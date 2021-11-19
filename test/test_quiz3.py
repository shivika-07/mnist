from sklearn import datasets
import numpy as np
from joblib import dump, load

def  test_digit_correct_0():
	digits = datasets.load_digits()
	data=digits.images; target=digits.target
	data = data.reshape((digits.images.shape[0], -1))
	res=np.where(target==0); op=res[0]
	clf=load("model1svm.joblib")
	pred=clf.predict(data[op])
	assert pred[0]==0

def  test_digit_correct_1():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==1); op=res[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[op])
        assert pred[0]==1

def  test_digit_correct_2():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((digits.images.shape[0], -1))
        res=np.where(target==2); op=res[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[op])
        #print(pred)
        assert pred[0]==2
def  test_digit_correct_3():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==3)
        k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==3
def  test_digit_correct_4():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==4)
        k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==4
def  test_digit_correct_5():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==5)
        k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[1]==5
def  test_digit_correct_6():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==6); k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==6
def  test_digit_correct_7():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==7)
        k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==7
def  test_digit_correct_8():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==8)
        k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==8
def  test_digit_correct_9():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==9); k=l[0]
        clf=load("model1svm.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==9



def  test_digit_correct_0dt():
	digits = datasets.load_digits()
	data=digits.images; target=digits.target
	data = data.reshape((data.shape[0], -1))
	l=np.where(target==0)
	k=l[0]
	clf=load("modeldecisiontree.joblib")
	pred=clf.predict(data[k])
	#print(pred)
	assert pred[0]==0
def  test_digit_correct_1dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==1); k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==1
def  test_digit_correct_2dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==2)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==2
def  test_digit_correct_3dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==3)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==3
def  test_digit_correct_4dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==4)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[0]==4
def  test_digit_correct_5dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==5)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        #print(pred)
        assert pred[1]==5
def  test_digit_correct_6dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==6)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==6
def  test_digit_correct_7dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==7)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==7
def  test_digit_correct_8dt():
        digits = datasets.load_digits()
        data=digits.images
        target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==8)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==8
def  test_digit_correct_9dt():
        digits = datasets.load_digits()
        data=digits.images; target=digits.target
        data = data.reshape((data.shape[0], -1))
        l=np.where(target==9)
        k=l[0]
        clf=load("modeldecisiontree.joblib")
        pred=clf.predict(data[k])
        assert pred[0]==9