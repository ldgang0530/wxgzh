#!/user/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import perception


def main():
    x = ([0.1, 0.2], [1, 1.5], [2, 2.2], [3, 4], [5, 7], [2.1, 1.2], [3.5, 3], [6, 5], [5.5, 4])
    y = [1, 1, 1, 1, 1, -1, -1, -1, -1]
    x_arr = np.array(x)
    y_arr = np.array(y)
    dataclass = perception.Perception(x_arr, y_arr, 1, 200)
    dataclass.perception()
    dataclass.display_margin()

    test_x1 = [600, 8]
    print("data:", test_x1, "class:", dataclass.classify(test_x1))
    test_x2 = [400, 3]
    print("data:", test_x2, "class:", dataclass.classify(test_x2))
    disp(x)

def disp(x):
    x_arr = np.array(x)
    x_arr0 = x_arr[x_arr[:, 0] > x_arr[:, 1]]
    x_arr1 = x_arr[x_arr[:, 1] > x_arr[:, 0]]
    x_axis0 = x_arr0[:, 0]
    y_axis0 = x_arr0[:, 1]
    x_axis1 = x_arr1[:, 0]
    y_axis1 = x_arr1[:, 1]
    plt.plot(x_axis0, y_axis0, 'bo', x_axis1, y_axis1, 'ro')


if __name__ == "__main__":
    main()
