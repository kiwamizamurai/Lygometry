import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

test = np.array([[6,4,0],[1,6,9],[12,7,0],[14,16,2],[7,7,2],[1,13,9],[2,14,10],[1,16,19],[6,14,1],[11,1,9]])
test.shape
initial = np.random.randn(10,2)


def dij(X,i, j, sigma_i):
    return (np.linalg.norm(X[i,:] - X[j,:])**2)/(2*((sigma_i)**2))
def pij(X, i, j, sigma_i):
    numerator = np.exp(-dij(X, i,  j, sigma_i))
    denominator = -1.0
    for k in xrange(X.shape[0]):
        denominator = denominator + np.exp(-dij(X, i,  k, sigma_i))
    return numerator/denominator
def qij(Y, i, j):
    numerator = np.exp(-np.linalg.norm(Y[i,:] - Y[j,:])**2)
    denominator = -1
    for k in xrange(Y.shape[0]):
        denominator = denominator + np.exp(-np.linalg.norm(Y[i,:] - Y[k,:])**2)
    return numerator/denominator
def costfunc(X, Y,sigma_i):
    C = 0
    x = X.shape[0]
    for m in xrange(x):
        for n in xrange(x):
            C += pij(X,m,n,sigma_i) * np.log( (pij(X,m,n,sigma_i)) / (qij(Y,m,n) ))
    return C
def gradient(X, Y, sigma_i):
    x = Y.shape[0]
    a = np.zeros((x,2))
    for i in xrange(x):
        for k in xrange(x):
            a[i] =   (Y[i,:]-Y[k,:])*(pij(X,i,k,sigma_i)-qij(Y,i,k)+pij(X,k,i,sigma_i)-qij(Y,k,i))
    return 2*a
def gradient_descent(func,X,learning_rate,iteration_max,initial,sigma_i):
    for i in xrange(iteration_max):
        #new = np.zeros((initial.shape[0],2))
        new = initial - learning_rate * gradient(X,initial,sigma_i)
        if (costfunc(X,initial,sigma_i) - costfunc(X,new,sigma_i)) < 1e-10:
            break
        initial = new
    return new
def sne(data,initial,sigma_i,learning_rate,iteration_max):
    ret = gradient_descent(costfunc(data,initial,sigma_i),data,learning_rate,iteration_max,initial,sigma_i)
    plt.scatter(ret[:,0],ret[:,1])
    #plt.savefig("/Users/Jin/Desktop/test_sne.png")



import os
import shutil
root, ext = os.path.splitext('/Users/Jin/Desktop/test_sne.png')
dir_path = root + '_testbox'
if os.path.isdir(dir_path):
    shutil.rmtree(dir_path)
    os.mkdir(dir_path)
else:
    os.mkdir(dir_path)

p = 0
for q in  [1.0,2.0,10.0]:
    sne(test,initial,q,0.5,100)
    file_name =  dir_path + '/' + str(p) + ext
    plt.savefig(file_name)
    p += 1
