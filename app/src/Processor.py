from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
import SVM
import KNN
import WSD

def runReW(data, timeWindow, batchSize):
    wsdActual = [0]*batchSize
    wsdPredicted = [0]*batchSize
    wsdTtimes = [0.0]*batchSize
    wsdCimes = [0.0]*batchSize
    wsdRmse = [0.0]*batchSize
    for i in range(batchSize):
        wsdActual[i], wsdPredicted[i], wsdTtimes[i], wsdCimes[i] = WSD.runReW(data, timeWindow)
        wsdRmse[i] = sqrt(mean_squared_error(wsdActual[i], wsdPredicted[i]))
    print('ReW: %.3f +- %.3f RMSE' % (np.mean(wsdRmse), np.std(wsdRmse)))
    print('ReW: %.3f +- %.3f train %.3f +- %.3f predict' % (np.mean(wsdTtimes), np.std(wsdTtimes), np.mean(wsdCimes), np.std(wsdCimes)))
    return wsdPredicted

def runCReW(data, timeWindow, batchSize):
    wsdActual = [0]*batchSize
    wsdPredicted = [0]*batchSize
    wsdTtimes = [0.0]*batchSize
    wsdCimes = [0.0]*batchSize
    wsdRmse = [0.0]*batchSize
    for i in range(batchSize):
        wsdActual[i], wsdPredicted[i], wsdTtimes[i], wsdCimes[i] = WSD.runCReW(data, timeWindow)
        wsdRmse[i] = sqrt(mean_squared_error(wsdActual[i], wsdPredicted[i]))
    print('CReW: %.3f +- %.3f RMSE' % (np.mean(wsdRmse), np.std(wsdRmse)))
    print('CReW: %.3f +- %.3f train %.3f +- %.3f predict' % (np.mean(wsdTtimes), np.std(wsdTtimes), np.mean(wsdCimes), np.std(wsdCimes)))
    return wsdPredicted

def runKNN(file, timeWindow, batchSize):
    knnActual = [0]*batchSize
    knnPredicted = [0]*batchSize
    knnTtimes = [0.0]*batchSize
    knnCimes = [0.0]*batchSize
    knnRmse = [0.0]*batchSize
    for i in range(batchSize):
        knnActual[i], knnPredicted[i], knnTtimes[i], knnCimes[i] = KNN.run(file)
        knnRmse[i] = sqrt(mean_squared_error(knnActual[i], knnPredicted[i]))
    print('KNN: %.3f +- %.3f RMSE' % (np.mean(knnRmse), np.std(knnRmse)))
    print('KNN: %.3f +- %.3f train %.3f +- %.3f predict' % (np.mean(knnTtimes), np.std(knnTtimes), np.mean(knnCimes), np.std(knnCimes)))
    return knnPredicted

def runSVM(data, timeWindow, batchSize):
    svmActual = [0]*batchSize
    svmPredicted = [0]*batchSize
    svmTtimes = [0.0]*batchSize
    svmCimes = [0.0]*batchSize
    svmRmse = [0.0]*batchSize
    for i in range(batchSize):
        svmActual[i], svmPredicted[i], svmTtimes[i], svmCimes[i] = SVM.run(data)
        svmRmse[i] = sqrt(mean_squared_error(svmActual[i], svmPredicted[i]))
    print('SVM: %.3f +- %.3f RMSE' % (np.mean(svmRmse), np.std(svmRmse)))
    print('SVM: %.3f +- %.3f train %.3f +- %.3f predict' % (np.mean(svmTtimes), np.std(svmTtimes), np.mean(svmCimes), np.std(svmCimes)))
    return svmPredicted