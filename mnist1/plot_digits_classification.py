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


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)

rescale_factors = [1]
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier

for rescale_factor in rescale_factors:
    gammaV = [0.0001,0.001,0.01,0.1,1]

    # trainaccuracylst = []
    # valaccuracylst = []
    # testaccuracylst = []
    model_candidates = []
    for gamma in gammaV:
        resized_images = []
        for d in digits.images:
            resized_images.append(rescale(d, rescale_factor, anti_aliasing=False))

        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_samples, -1))

        clf = svm.SVC(gamma=gamma)

        # Split data into 50% train and 50% test subsets
        X_train, X_val, y_train, y_val = train_test_split(
            data, digits.target, test_size=0.3, shuffle=False)

        X_val, X_test, y_val, y_test = train_test_split(
            X_val, y_val, test_size=0.5, shuffle=False)
        

        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        predicted_valid = clf.predict(X_val)
        acc_valid = metrics.accuracy_score(y_pred=predicted_valid, y_true=y_val)
        f1_valid = metrics.f1_score(
            y_pred=predicted_valid, y_true=y_val, average="macro"
        )

        if acc_valid < 0.11:
                print("Skipping for {}".format(gamma))
                continue
        
        candidate = {
                "acc_valid": acc_valid,
                "f1_valid": f1_valid,
                "gamma": gamma,
            }
        model_candidates.append(candidate)
        max_valid_f1_model_candidate = max(
            model_candidates, key=lambda x: x["f1_valid"]
        )

        predicted = clf.predict(X_test)

        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
        print(
            "{}x{}\t{}\t{:.3f}\t{:.3f}".format(
                resized_images[0].shape[0],
                resized_images[0].shape[1],
                max_valid_f1_model_candidate["gamma"],
                acc,
                f1,
            )
        )

