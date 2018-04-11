import numpy as np
import matplotlib.pyplot as plt
import pdb
import time


def d_ij(X,i,j,sigma):
    return ((np.linalg.norm(X[i,:]-X[j,:])**2)/(2*(sigma**2)))


def p_ij(X,i,j,sigma):
    numerator = np.exp(-d_ij(X,i,j,sigma))
    denominator = 0
    for f in range(X.shape[0]):
        denominator = denominator + np.exp(-d_ij(X,i,f,sigma))
    denominator = denominator - 1
    return numerator/denominator

def q_ij(Y,i,j,sigma):
    numerator = np.exp(-np.linalg.norm(Y[i,:]-Y[j,:])**2)
    denominator = 0
    for f in range(Y.shape[0]):
        denominator = denominator + np.exp(-(np.linalg.norm(Y[i,:]-Y[j,:])**2))
    denominator = denominator - 1
    return numerator/denominator


def gradient_fi(Y,i):
    for j in xrange(Y.shape[0]):
        sumation = sumation + (Y[i,:] - Y[j,:])*(p_ij(X,i,j,2) - q_ij(Y,i,j,2) + p_ij(X,j,i,2)- q_ij(Y,j,i,2) )
    return 2*sumation

def gradient_f(Y):
    x = gradient_fi(Y,1)
    for d in xrange(Y.shape[0]):
        np.c_((x,gradient_fi(Y,d)))
    return x

def gd_method(gradient_f, init_pos, learning_rate):
    eps = 1e-10

    init_pos = np.array(init_pos)
    pos = init_pos
    pos_history = [init_pos]
    iteration_max = 1000

    for i in range(iteration_max):

        pos_new = pos - learning_rate * gradient_f(pos)

        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)
    return (pos, np.array(pos_history))



'''
For show
'''

def unkoman(a,b):
    d =  a +b
    start_time = time.time()
    for k in range(1000):
        unko = np.random.normal(0,1, 100)
        if np.mod(k, 100)==0 :
            print("%s th iteration complete"%k)
    end_time = time.time()
    print("elapsed time %s secs"%(end_time - start_time)) 

    return unko
