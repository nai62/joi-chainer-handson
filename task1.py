import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np


def draw_figure(ax, w, b, xlim=(15, 35), ylim=(0, 12), x_range=(17, 33)):
    """グラフを描画するためのヘルパー関数"""

    ax.scatter(x, t, color='blue', linewidths='1')
    ax.plot(x_range, w * x_range + b, color='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def train(x, t, initial_w, initial_b, epoch, alpha, show_figure):
    """学習を行う関数"""

    x = np.array(x, dtype=np.float32)
    t = np.array(t, dtype=np.float32)

    # 重みの初期化
    w = chainer.Variable(np.array(initial_w, dtype=np.float32))
    b = chainer.Variable(np.array(initial_b, dtype=np.float32))

    if show_figure:
        fig = plt.figure(figsize=(16, 9))

    for i in range(epoch):
        # 勾配をクリア
        w.cleargrad()
        b.cleargrad()

        # 家賃の予測値 y を計算
        y = w * x + b
        # 損失を計算
        loss = F.sum(F.squared_error(y, t))

        # グラフの描画
        if show_figure:
            ax = fig.add_subplot((epoch + 4) // 5, 5, i + 1)
            draw_figure(ax, w.array, b.array)

        # 自動微分により w, b の勾配を計算
        loss.backward()

        # 勾配を用いて w, b の値を更新
        w.array -= alpha * w.grad
        b.array -= alpha * b.grad

    if show_figure:
        plt.show()
    return w, b


x = [20, 25, 30]  # 入力（部屋の広さ）
t = [4, 7, 9]  # 教師データ（家賃）

initial_w = 0.4  # 傾きの初期値
initial_b = -6  # 切片の初期値
epoch = 20  # エポック数（データを何周するか）
alpha = 0.00004  # 学習率

# 学習
w, b = train(x, t, initial_w, initial_b, epoch, alpha, show_figure=True)
print(f'w = {w}, b = {b}')

# 最終結果の表示
ax = plt.figure().add_subplot(111)
draw_figure(ax, w.array, b.array)
plt.show()
