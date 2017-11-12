import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(10,10)

def main():
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w","brown","coral"]
    for i,c in zip(range(len(X[0,:])),colorlist):
        for j in range(len(X[0,:])):
            plt.scatter(j,P_ij(X,i,j),color=c,s=10)
    plt.savefig('../../dataset/gene/p_ij.png')
if __name__ == '__main__':
    main()
