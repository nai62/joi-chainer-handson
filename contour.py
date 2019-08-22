import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np


def loss(w, b):
    x = np.array([20, 25, 30])
    t = np.array([4, 7, 9])
    y = w * x + b
    error = y - t
    return (error * error).sum(axis=-1)


def main():
    ws, bs = np.meshgrid(np.linspace(0, 1, 100), np.linspace(-20, 5, 100))
    losses = loss(ws[:, :, None], bs[:, :, None])
    print(losses.shape)

    fig = plt.figure(figsize=(16, 8))

    # 等高線を作成する。
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(ws, bs, np.log(losses), levels=100)
    print(type(contour))  # <class 'matplotlib.contour.QuadContourSet'>
    ax1.set_xlabel('w', fontsize=18)
    ax1.set_ylabel('b', fontsize=18, rotation=0)

    # 3D グラフを作成する。
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(ws, bs, losses, cmap=cm.coolwarm, antialiased=True)
    ax2.set_xlabel('w', fontsize=18)
    ax2.set_ylabel('b', fontsize=18, rotation=0)

    plt.show()


if __name__ == '__main__':
    main()
