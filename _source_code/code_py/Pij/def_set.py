import numpy as np


def d_ij_2(X,i,j):
    delta_i = 1000
    d = np.exp(-(np.sum((X[:,i]-X[:,j])**2))/(2*(delta_i**2)))
    return d

def P_ij(X,i,j):
    h = -1
    for j_2 in range(len(X[0,:])-1):
        d = d_ij_2(X,i,j_2)
        h = h + d
    d = d_ij_2(X,i,j)
    answer = d/h
    return answer
