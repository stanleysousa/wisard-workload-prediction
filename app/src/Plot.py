from datetime import datetime
import matplotlib.pyplot as plt

def serie(data, timeWindow):
    y = data[:, 1].astype(int)
    tmp = data[:, 0]
    x = []
    refdate = datetime(1995, 7, 1, 0, 0, 0)
    for i in range(len(tmp)):
        d = datetime.fromisoformat(tmp[i])-refdate
        v = (d.total_seconds()//(timeWindow*60)) if timeWindow > 0 else d.total_seconds()
        x.append(int(v))
    plt.plot(x, y)
    plt.xlabel('Tempo (horas desde o primeiro registro)')
    plt.ylabel('Requisições')
    plt.show()

def zoom(data, reWPredicted, cReWPredicted, knn, svm, timeWindow):
    size = int(len(data) * 0.67)

    tmp = data[size+1:len(data), 0]
    x = []
    refdate = datetime(1995, 7, 1, 0, 0, 0)
    for i in range(len(tmp)):
        d = datetime.fromisoformat(tmp[i])-refdate
        v = (d.total_seconds()//(timeWindow*60)) if timeWindow > 0 else d.total_seconds()
        x.append(int(v))

    y = data[size+1:len(data), 1].astype(int)

    plt.plot(x[100: 150],   y[100: 150], '-', color='black' ,label='Atual', linewidth=2)
    plt.plot(x[100: 150], reWPredicted[100: 150], '-', color='olive' ,label='ReW')
    plt.plot(x[100: 150], cReWPredicted[100: 150], '-', color='blue' ,label='CReW')
    plt.plot(x[100: 150], knn[100: 150], '-', color='purple' ,label='KNN')
    plt.plot(x[100: 150], svm[100: 150], '-', color='green' ,label='LinearSVM')

    plt.xlabel('Intervalos de '+str(timeWindow)+ 'minutos')
    plt.ylabel('Requisições')
    plt.ylim(ymin=0)
    plt.legend()
    plt.show()