import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,10))
def plot_mesh(NL, A, B):
    plt.scatter(NL[:,0], NL[:,1], c='black')
    for a, b in zip(NL[0:,0], NL[0:,1]):
        plt.vlines(a, ymin=1, ymax=A, color="b", alpha=1)
        plt.hlines(b, xmin=1, xmax=B, color="c", alpha=1)
        # plt.savefig('saved_figure.png')
        
def plot_crack(cracktip_1, cracktip_2):
    plt.scatter(cracktip_1, cracktip_2, c="b")
    plt.plot(cracktip_1, cracktip_2, linewidth=3, c="r")
    # plt.savefig('saved_figure.png')
    
def plot_sub_elements(zeta):
    y = zeta[0:,1]
    x = zeta[0:,0]
    y_max = np.max(y)
    y_min = np.min(y)
    x_max = np.max(x)
    x_min = np.min(x)
    plt.scatter(zeta[:,0], zeta[:,1], c='red')
    for a, b in zip(zeta[0:,0], zeta[0:,1]):
        plt.vlines(a, ymin=y_min, ymax=y_max, color="green", alpha=1)
        plt.hlines(b, xmin=x_min, xmax=x_max, color="green", alpha=1)
        # plt.savefig('saved_figure.png')   

def plot_sub_nodes(eta):
    plt.scatter(eta[:,0], eta[:,1], c='black',s=50)
    # plt.savefig('saved_figure.png')
    
