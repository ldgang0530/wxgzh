#!/user/bin/env python
import Logistic


def load_dataset():
    xdata =([5, 6], [4, 9], [6, 7], [12, 15], [3, 7], [4, 9], [1, 8], [2, 9], [6, 12], [10, 13],
            [9, 1], [8, 4], [15, 12], [7, 3], [9, 5], [13, 10], [6, 1], [14, 12], [5, 1], [16, 13]
            )
    ydata = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return xdata, ydata


def main():
    xdata, ydata = load_dataset()
    logistic = Logistic.Logistic(xdata, ydata, 0.3, 0.05, 50)
    logistic.train()
    print("Right Rate:", logistic.right_rate())
    test_x1 = ([2, 9])
    print("Predict class:", logistic.classify(test_x1))


if __name__ == "__main__":
    main()
