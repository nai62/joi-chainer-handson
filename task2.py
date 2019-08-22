import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np


def train(x, t, dim_u, epoch, alpha, show_loss=False):
    """学習を行う関数。学習した層（l1, l2）を返す。"""

    x = np.array(x, dtype=np.float32)
    t = np.array(t, dtype=np.float32)[:, np.newaxis]

    # 層を初期化（重みはランダム）
    l1 = L.Linear(dim_u)  # 中間層 u の次元 == dim_u
    l2 = L.Linear(1)  # 出力層 y の次元 == 1

    losses = []
    for i in range(epoch):
        # 勾配をクリア
        l1.cleargrads()
        l2.cleargrads()

        # 予測値 y を計算
        u = l1(x)
        h = F.relu(u)
        y = l2(h)

        # 損失を計算
        loss = F.sum(F.squared_error(y, t))
        losses.append(loss.array)

        # 自動微分によりそれぞれの層の勾配を計算
        loss.backward()

        # 勾配を用いて重みを更新
        l1.W.array -= alpha * l1.W.grad
        l1.b.array -= alpha * l1.b.grad
        l2.W.array -= alpha * l2.W.grad
        l2.b.array -= alpha * l2.b.grad

    if show_loss:
        plt.plot(range(epoch), losses)
        plt.show()

    return l1, l2


x = [[0, 0], [0, 1], [1, 0], [1, 1]]  # 入力 (x1, x2)
t = [0, 1, 1, 0]  # 教師データ (x1 ^ x2)
# x = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]  # 0~2どうしの排他的論理和
# t = [0, 1, 2, 1, 0, 3, 2, 3, 0]  # その答え

dim_u = 2  # 中間層の次元数
epoch = 100  # エポック数
alpha = 0.05  # 学習率

# 学習
l1, l2 = train(x, t, dim_u, epoch, alpha, show_loss=False)

# 最終結果の表示
u = l1(np.array(x, dtype=np.float32))
h = F.relu(u)
y = l2(h)
for (x1, x2), y1 in zip(x, y.array):
    print(f'{x1} と {x2} の排他的論理和は {y1} です')
