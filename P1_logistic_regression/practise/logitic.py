# Logistic regression

# import lib
import numpy as np
import matplotlib.pyplot as plt

# def load data
def loadData(filename):
    file = open(filename)
    x = []
    y = []
    for line in file.readlines():
        line = line.strip().split()
        x.append([1,float(line[0]), float(line[1])])
        y.append(float(line[2]))
    xmat = np.mat(x)
    ymat = np.mat(y).T
    file.close()
    return xmat, ymat
# w calc
def w_calc(xmat, ymat,alpha = 0.001, maxIter = 10001):
    #W init
    W = np.mat(np.random.rand(3,1))  #最好不要0
    print(W)
    #W update
    for i in range(maxIter):
        H = 1 / (1 + np.exp(-xmat * W))
        dw = xmat.T * (H - ymat) # shape(3,1)
        W -= alpha * dw
    return W

# implement
xmat, ymat = loadData('TestSet.txt')
W = w_calc(xmat,ymat)
# show
w0 = W[0,0]
w1 = W[1,0]
w2 = W[2,0]
plotx1 = np.arange(-3.0, 3.0, 0.01)
plotx2 = -w0/w2 - w1/w2 * plotx1
plt.plot(plotx1,plotx2, c='r', label = 'decision boundary')


plt.scatter(xmat[:,1][ymat==0].A, xmat[:,2][ymat==0].A, marker='^', s=150, label = 'label=0')
plt.scatter(xmat[:,1][ymat==1].A, xmat[:,2][ymat==1].A, s=150, label = 'label=1')
plt.grid()
plt.legend()
plt.show()