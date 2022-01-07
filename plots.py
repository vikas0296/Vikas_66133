import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(10)
# plt.title('2D_UNIFORM_MESH')
# plt.ylabel('Y-AXIS')
# plt.xlabel('X-AXIS')
# ax = plt.subplot(111)

def plot_mesh(NL, A, B, X):
    plt.scatter(NL[:,0], NL[:,1], c='black', s=50, label = "Node")
    plt.legend(loc="upper right")

    for a, b, in zip(NL[0:,0], NL[0:,1]):
        plt.vlines(a, ymin=1, ymax=A, color="b", alpha=1)
        plt.hlines(b, xmin=1, xmax=B, color="c", alpha=1)
        # plt.savefig('saved_figure.png')
    plt.ylabel('Y-AXIS')
    plt.xlabel('X-AXIS')
    plt.title(f'2D_UNIFORM_MESH of size {X}x{X}')


def plot_crack(cracktip_1, cracktip_2):
    plt.scatter(cracktip_1, cracktip_2, c="b")
    plt.plot(cracktip_1, cracktip_2, linewidth=3, c="r", label = "Crack")
    plt.legend(loc="upper right")
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

def DOFs(alpha):
    Z = np.asarray(alpha)
    plt.scatter(Z[:,0], Z[:,1], c='Blue', s=200,label = "DOFs")
    plt.legend(loc="upper right")

def force(charlie):
    for i in charlie:
        plt.scatter(i[0], i[1], c='Orange', alpha=1, s=200, label = "Force(N)")
    plt.legend(loc="upper right")
# plt.arrow(1, 1, 0, 2, color="green")

def extra_dofs(omega):
    x = np.asarray(omega)
    plt.scatter(x[:,0], x[:, 1], c='black', marker='s', s = 100, label = "Tip_enriched")
    plt.legend(loc="upper right")

def deformation_plots(NL, displacement_vector):
    Vegeta = displacement_vector[0:len(displacement_vector):2]
    goku = displacement_vector[1:len(displacement_vector):2]
    Beerus = []
    for i,j,k in zip(NL, Vegeta, goku):
        gohan = [0,0]
        gohan[0] = i[0] + j
        gohan[1] = i[1] + k
        Beerus.append(gohan)

    # print(len(Beerus))
    plt.figure(figsize=(10,10))
    for i in Beerus:
        plt.scatter(round(i[0],3), round(i[1],3), c='red')

























