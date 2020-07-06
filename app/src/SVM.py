import time
from datetime import datetime
import numpy as np
from sklearn.svm import LinearSVR

def _preProccess(data):
    y = data[:, 1].astype(float)
    tmp = data[:, 0]
    x = []
    for i in range(len(tmp)):
        d = datetime.fromisoformat(tmp[i])
        v = d.hour*60+d.minute
        x.append([d.weekday(), float(v)])
    return y, x

def run(data):
    y, x = _preProccess(data)
    size = int(len(data) * 0.67)
    yTrain = y[:size]
    xTrain = x[:size]
    yTest = y[size+1:len(data)]
    xTest = x[size+1:len(data)]

    model = LinearSVR(epsilon=2.0, tol=0.001, C=1.0, loss='squared_epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=False, verbose=0, random_state=None, max_iter=1000)

    # fit the model with the training data
    tStart = time.time()
    model.fit(xTrain,yTrain)
    ttime = time.time()-tStart

    # predict the target on the test dataset
    cStart = time.time()
    predicted = model.predict(xTest)
    ctime = time.time()-cStart

    actual = yTest
    return actual, predicted, ttime, ctime