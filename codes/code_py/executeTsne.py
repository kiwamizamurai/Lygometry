import argparse
import numpy as np
from matplotlib import pyplot as plt
import util
import sys
import os
import pdb

figpath = '../figures'



def main(args):
    print("UNKO")

    parameter = args.parameter
    unko = util.unkoman(1,parameter)

    plt.hist(unko)

    figname= os.path.join(figpath, "unko.png")
    plt.savefig(figname)
    os.system("open -a Finder %s"%figpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parameter',  '-p', type=float, default = 10)
    args = parser.parse_args()

    main(args)
