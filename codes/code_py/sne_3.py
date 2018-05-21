import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import seaborn as sns
from progressbar import ProgressBar
from tqdm import tqdm

sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

__version__ =  '1.0.0'








class SNE(object):
    def __init__(self, X, dim , eta, n_iter, sigma, method):
        self.X = X
        self.n = X.shape[0]
        self.dim = dim
        self.eta = eta
        self.iter = n_iter
        self.sigma = sigma
        self.method = method

## Normalize datasets ---------------------------------------------------------------------------------------------

    def normalize(self):
        self.X = (self.X - np.mean(self.X))/np.std(self.X)

## Initialize y ---------------------------------------------------------------------------------------------

    def initialize(self):
        if self.dim == 2:
            print "2-dimensional space"
            self.y = np.random.randn(self.n, self.dim)
        elif self.dim == 3:
            print "3-dimensional space"
            self.y = np.random.randn(self.n, self.dim)

## Make D_ij ---------------------------------------------------------------------------------------------

    def d_ij(self, i, j):
        return ((np.linalg.norm(self.X[i,:] - self.X[j,:])**2)/(2*(self.sigma**2)))

## Make P_ij ---------------------------------------------------------------------------------------------

    def p_ij(self, i, j):
        if i == j:
            return 0
        numerator = np.exp(-self.d_ij(i, j))
        denominator = 0
        for f in range(self.n):
            denominator = denominator + np.exp(-self.d_ij(i, f))
        denominator = denominator - 1
        return numerator/denominator

## Make Q_ij ---------------------------------------------------------------------------------------------

    def q_ij(self, i, j):
        if i == j:
            return 0
        numerator = np.exp(-np.linalg.norm(self.y[i,:] - self.y[j,:])**2)
        denominator = 0
        for f in range(self.n):
            denominator = denominator + np.exp(-(np.linalg.norm(self.y[i,:] - self.y[f,:])**2))
        denominator = denominator - 1
        return numerator/denominator

## Calculate cost-function --------------------------------------------------------------------------------

    def costfunction(self):
        cost = 0
        for i in tqdm(range(self.n), desc='cost'):
            for j in range(self.n):
                # this condition is very important
                if i == j:
                    continue
                else:
                    val = self.p_ij(i, j)/self.q_ij(i, j)
                    left = np.log(val)
                    cost = cost + (self.p_ij(i, j) * left)
        return cost

## Calculate gradient --------------------------------------------------------------------------------

    def gradient(self):
        y = np.zeros_like(self.y)
        for i in tqdm(range(0, self.n), desc='grad'):
            y_i = np.zeros_like(self.y[i])
            for j in range(0, self.n):
                rightinsigma = self.p_ij(i, j)-self.q_ij(i, j)+self.p_ij(j, i)-self.q_ij( j, i)
                y_i += (self.y[i] - self.y[j])*rightinsigma
            y[i] = 2*y_i
        return y

## update --------------------------------------------------------------------------------

    def update(self):
        self.epoch = []
        self.cost = []

        pre_cost = self.costfunction()
        for step in xrange(self.iter):
            print("Step: {}".format(step))
            print("Present cost : {}".format(pre_cost))
            self.epoch.append(step)
            self.cost.append(pre_cost)
            self.y = self.y - self.eta * self.gradient()
            post_cost = self.costfunction()
            diff = abs(pre_cost - post_cost)
            pre_cost = post_cost
            if diff < 0.01:
                print("Converged")
                return self.y
        print("Unconverge")

## visualize cost-value ------------------------------------------------------------------

    def costplot(self):
        plt.figure(figsize=(6,6))
        plt.plot(self.epoch, self.cost)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title("cost_function")
        plt.show()

## visualize for 2 dimensional space ------------------------------------------------------------------

    def plot(self):
        plt.figure(figsize=(6,6))
        plt.scatter(self.y[0:29, 0], self.y[0:29, 1], c = "blue")
        plt.scatter(self.y[30:60, 0], self.y[30:60, 1], c = "red")
        plt.title("After sne")
        plt.show()









def main():
    data = np.loadtxt(args.file,delimiter=",")
    data = data[0:20,0:3]
    print("The shape of data : ", data.shape)
    sneimp = SNE(
    X = data,
    dim = args.dim,
    eta = args.eta,
    n_iter = args.iteration,
    sigma = args.sigma,
    method = args.gradient_method
    )
    sneimp.normalize()
    sneimp.initialize()
    sneimp.update()
    sneimp.costplot()
    sneimp.plot()







if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    prog = "SNE",
    usage = """
    This requires only datasets that you want to reduct dimension.
    You shold give file name or path like;
    $python3 sne_3.py -f test.csv
    """,
    description = """
    This program supports only Mac and Python3.
    I am going to improve this as to adapt to any other OS and version.
    """,
    epilog = "end"
    )

    parser.add_argument('-v', '--version', action='version',
                        version=('%(prog)s ' + __version__  ))
    parser.add_argument("-f", "--file", metavar = "DATA",
                         type = str, help = "high-dimensional datasets", required = True)
    parser.add_argument("-d", "--dim", metavar = "dim",
                         type = int, default = 2, help = "R^2 or R^3",
                         choices = [2,3]
                         )
    parser.add_argument('-e', '--eta', metavar = "eta",
                        type = float, default = 0.1, help = "default value: 0.1")
    parser.add_argument('-i', '--iteration', metavar = "iteration",
                        type = int, default = 1000,  help = "default value: 1000")
    parser.add_argument('-s', '--sigma', metavar = "sigma",
                        type = int, default = 1, help = "default value: 1")
    parser.add_argument('-gm', '--gradient_method', metavar = "gradient_method",
                        type = int, default = 0, help =
                        """
                        0: Gradient descent\n
                        1: Stochastic gradient method\n
                        """,
                        choices = [0,1]
                        )

    args = parser.parse_args()
    main()
