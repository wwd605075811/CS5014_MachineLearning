#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def one_bezier_curve(a, b, t):
    return (1 - t) * a + t * b


# xs表示原始数据
# n表示阶数
# k表示索引
def n_bezier_curve(xs, n, k, t):
    if n == 1:
        return one_bezier_curve(xs[k], xs[k + 1], t)
    else:
        return (1 - t) * n_bezier_curve(xs, n - 1, k, t) + t * n_bezier_curve(xs, n - 1, k + 1, t)


def bezier_curve(xs, ys, num, b_xs, b_ys):
    n = len(xs) - 1
    t_step = 1.0 / (num - 1)
    t = np.arange(0.0, 1 + t_step, t_step)
    for each in t:
        b_xs.append(n_bezier_curve(xs, n, 0, each))
        b_ys.append(n_bezier_curve(ys, n, 0, each))


def main():
    xs = [0, 2, 5, 10, 15, 20]
    ys = [0, 6, 10, 0, 5, 5]
    num = 100
    b_xs = []
    b_ys = []
    bezier_curve(xs, ys, num, b_xs, b_ys)
    plt.figure()
    plt.plot(b_xs, b_ys)
    plt.plot(xs, ys)
    plt.show()


if __name__ == "__main__":
    main()
