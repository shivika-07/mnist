"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
from itertools import accumulate
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from skimage.transform import rescale
import numpy as np
import os
from joblib import dump, load
from utils import *

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)

rescale_factors = [1]
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier

for rescale_factor in rescale_factors:
    gammaV = [10 ** exponent for exponent in range(-7, 0)]

    model_candidates = []
    for gamma in gammaV:
        resized_images = preprocess(digits.images,rescale_factor)
        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_samples, -1))

        clf = svm.SVC(gamma=gamma)

        # Split data into 50% train and 50% test subsets
        test_size,val_size = 0.15, 0.15
        X_train, X_test, X_val,y_train,y_test,y_val = create_split(data, digits.target,test_size, val_size)       

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        metric_valid = test(clf, X_val, y_val)

        if metric_valid['acc'] < 0.11:
                print("Skipping for {}".format(gamma))
                continue
        
        candidate = {
                "acc_valid": metric_valid['acc'],
                "f1_valid": metric_valid['f1'],
                "gamma": gamma,
            }
        model_candidates.append(candidate)

        # cwd = os.getcwd()
        # print(cwd)
        output_folder = "./mnist1/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, val_size, rescale_factor, gamma)
        os.mkdir(output_folder)
        dump(clf, os.path.join(output_folder,"model.joblib"))

    max_valid_f1_model_candidate = max(model_candidates, key=lambda x: x["f1_valid"])
    best_model_folder="./mnist1/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, val_size, rescale_factor, max_valid_f1_model_candidate['gamma']
            )
    metric = test(clf, X_test, y_test)
    print(
        "{}x{}\t{}\t{:.3f}\t{:.3f}".format(
            resized_images[0].shape[0],
            resized_images[0].shape[1],
            max_valid_f1_model_candidate["gamma"],
            metric['acc'],
            metric['f1'],
        )
    )

