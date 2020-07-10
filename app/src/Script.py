import Parser
import Processor
import Plot
import numpy as np

#Parse log files
julLogFile = "../data/in/access_log_Jul95"
augLogFile = "../data/in/access_log_Aug95"
#Parser.proccessLog(julLogFile, augLogFile)

#Load data
timeWindow = 60
batchSize = 10
file = "../data/out/data_"+str(timeWindow)+"min.csv"
data = np.loadtxt(file, dtype=str, delimiter=",", skiprows=1)

reWPredicted = Processor.runReW(data, timeWindow, batchSize)
cReWPredicted = Processor.runCReW(data, timeWindow, batchSize)
knnPredicted = Processor.runKNN(file, timeWindow, batchSize)
svmPredicted = Processor.runSVM(data, timeWindow, batchSize)

Plot.zoom(data, reWPredicted[0], cReWPredicted[0],knnPredicted[0], svmPredicted[0], timeWindow)

