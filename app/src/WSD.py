import time
from datetime import datetime
from datetime import timedelta
import wisardpkg as wp
import numpy as np


def _preProccess(data, timeWindow):
    y = data[:, 1].astype(float)
    tmp = data[:, 0]
    x = []
    for i in range(len(tmp)):
        d = datetime.fromisoformat(tmp[i])
        v = str(d.hour*360+d.minute*60+d.second)+str(d.weekday())
        x.append(float(v))
    termo = wp.SimpleThermometer(100, 1.0, max(x))
    for i in range(len(x)):
        x[i] = termo.transform([x[i]])
    return y, x

def runReW(data, timeWindow):
    y, x = _preProccess(data, timeWindow)

    size = int(len(data) * 0.67)
    yTrain = y[:size]
    xTrain = x[:size]
    yTest = y[size+1:len(data)]
    xTest = x[size+1:len(data)]

    model = wp.RegressionWisard(10, completeAddressing=False, orderedMapping=False, mean=wp.SimpleMean(), minZero=0, minOne=0)
    tStart = time.time()
    for i in range(len(xTrain)):
        model.train(xTrain[i], yTrain[i])
    ttime = time.time()-tStart

    predicted = []
    cStart = time.time()
    for i in range(len(xTest)):
        predicted.append(model.predict(xTest[i]))
    ctime = time.time()-cStart

    actual = yTest
    return actual, predicted, ttime, ctime

def runCReW(data, timeWindow):
    y, x = _preProccess(data, timeWindow)

    size = int(len(data) * 0.67)
    yTrain = y[:size]
    xTrain = x[:size]
    yTest = y[size+1:len(data)]
    xTest = x[size+1:len(data)]

    model = wp.ClusRegressionWisard(10, 0, 5, 75)
    tStart = time.time()
    for i in range(len(xTrain)):
        model.train(xTrain[i], yTrain[i])
    ttime = time.time()-tStart

    predicted = []
    cStart = time.time()
    for i in range(len(xTest)):
        predicted.append(model.predict(xTest[i]))
    ctime = time.time()-cStart

    actual = yTest
    return actual, predicted, ttime, ctime