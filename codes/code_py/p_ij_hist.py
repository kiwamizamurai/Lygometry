
import numpy as np
import matplotlib.pyplot as plt
from def_set import P_ij
from def_set import d_ij_2

X = np.random.randn(10,100)

def main():
    a = [P_ij(X,0,j) for j in range(len(X[0,:]))]
    plt.hist(a)
    plt.savefig('../../dataset/gene/p_ij_hist.png')
    
if __name__ == '__main__':
    main()
