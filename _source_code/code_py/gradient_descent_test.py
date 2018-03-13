import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

test = np.random.randn(100,30)
initial = np.random.randn(100,2)

def dij(X,i, j, sigma_i):
    return (np.linalg.norm(X[i,:] - X[j,:])**2)/2*((sigma_i)**2)

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
        new = np.zeros((initial.shape[0],2))
        new = initial - learning_rate * gradient(X,initial,sigma_i)
        if (costfunc(X,initial,sigma_i) - costfunc(X,new,sigma_i)) < 1e-10:
            break
        initial = new
    return new

gradient_descent(costfunc(test,initial,2),test,0.5,1000,initial,2.0)
