from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

X, Y = make_moons(random_state=42, n_samples=(50, 450), noise=0.25)

#Organize/Plot the data
fig = plt.figure()
ax = fig.add_subplot()
set1 = [] #
set2 = []
for i in range(len(Y)):
    if Y[i] == 1:
        set1.append(X[i])
    else:
        set2.append(X[i])

set1 = np.asarray(set1)
set2 = np.asarray(set2)
ax.scatter(set1[:,0], set1[:,1])
ax.scatter(set2[:,0], set2[:,1])

#Split data
trainPercentage = 0.7
X = np.asarray(X)
Y = np.asarray(Y)

trainSet = X[0:int(len(X)*trainPercentage)]
trainSetLabel = Y[0:int(len(Y)*trainPercentage)]

testSet = X[int(len(X)*trainPercentage):len(X)]
testSetLabel = Y[int(len(Y)*trainPercentage):len(Y)]

#https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py
clf = QuadraticDiscriminantAnalysis()
clf2 = clf.fit(trainSet, trainSetLabel)

#see results
#https://stackabuse.com/bytes/plot-decision-boundaries-using-python-and-scikit-learn/
disp = DecisionBoundaryDisplay.from_estimator(clf2,
                                              trainSet, 
                                              response_method="predict",
                                              xlabel="X Label", ylabel="Y Label",
                                              alpha=0.5, 
                                              cmap=plt.cm.coolwarm)

disp.ax_.scatter(trainSet[:, 0], trainSet[:, 1], 
                 c=trainSetLabel, edgecolor="k",
                 cmap=plt.cm.coolwarm)

plt.show()

