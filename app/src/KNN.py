import time
from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def _preProccess(data):
    y = data.iloc[:, 1]
    tmp = data.iloc[:, 0]
    x = []
    for i in range(len(tmp)):
        if i==0: continue #skip the header
        d = datetime.fromisoformat(tmp[i])
        v = d.hour*60+d.minute
        x.append([d.weekday(), float(v)])
    x = pd.DataFrame(x, columns = ['WeekDay' , 'EncodedTime'])
    return y, x

def run(file):
    data = pd.read_csv(file)

    y, x = _preProccess(data)
    size = int(len(data) * 0.67)
    yTrain = y.iloc[:size]
    xTrain = x.iloc[:size]
    yTest = y.iloc[size+1:len(data)]
    xTest = x.iloc[size:len(data)]

    model = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

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