from datetime import datetime
from datetime import timedelta

def _processLine(line):
    left = line.find('[', 0, len(line))+1
    right = line.find(']', 0, len(line))-6 #removes seconds and timezone
    if not line[left : right]:
        return None
    try:
        return datetime.strptime(line[left : right], '%d/%b/%Y:%H:%M:%S')
    except:
        return None

def _processFile(file, timeWindow):
    dates = []
    datesDic= dict()
    with open(file, "r", errors='ignore') as f:
        for line in f:
            date = _processLine(line)
            #skip the line if it does not contain a valid datetime
            if date is None:
                print(line)
                continue
            else:
                #increment
                dates.append(date)
    current = dates[0]
    window = dates[0]+timedelta(minutes=timeWindow)
    datesDic[current] = 0
    for d in dates:
        if d <= window:
            datesDic[current] = datesDic[current]+1
        else:
            current = d
            window = d+timedelta(minutes=timeWindow)
            datesDic[current] = 1
    return datesDic

def _read(julFile, augFile, timeWindow):
    julDict = _processFile(julFile, timeWindow)
    augDict = _processFile(augFile, timeWindow)

    data = []
    for entry in julDict:
        data.append([entry.isoformat(), julDict[entry]])
    for entry in augDict:
        data.append([entry.isoformat(), augDict[entry]])

    return data

def _write(filePath, data):
    with open(filePath, "w") as f:
        print('Datetime, Count', file=f)
        for row in data:
            print(row[0]+', '+str(row[1]), file=f)

def _proccessTimeWindow(julFile, augFile, timeWindow):
    outputFile = "../data/out/data_"+str(timeWindow)+"min.csv"
    data = _read(julFile, augFile, timeWindow)
    _write(outputFile, data)

def proccessLog(julFile, augFile):
    _proccessTimeWindow(julFile, augFile, 1)
    _proccessTimeWindow(julFile, augFile, 5)
    _proccessTimeWindow(julFile, augFile, 15)
    _proccessTimeWindow(julFile, augFile, 30)
    _proccessTimeWindow(julFile, augFile, 60)