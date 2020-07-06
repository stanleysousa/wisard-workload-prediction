import Parser
import Processor
import Plot
import numpy as np

#Parse log files
julLogFile = "../data/in/access_log_Jul95"
augLogFile = "../data/in/access_log_Aug95"
#Parser.proccessLog(julLogFile, augLogFile)

#Load data
timeWindow = 30
batchSize = 10
file = "../data/out/data_"+str(timeWindow)+"min.csv"
data = np.loadtxt(file, dtype=str, delimiter=",", skiprows=1)

wsdPredicted = Processor.runWSD(data, timeWindow, batchSize)
knnPredicted = Processor.runKNN(file, timeWindow, batchSize)
svmPredicted = Processor.runSVM(data, timeWindow, batchSize)

Plot.zoom(data, wsdPredicted[0], knnPredicted[0], svmPredicted[0], timeWindow)

