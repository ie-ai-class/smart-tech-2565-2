import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
# Decision surface


def plot_ds(clfs, names, X_train, y_train, X_plot, y_plot):
    all_clf = clfs
    clf_labels = names
    X_test = X_plot
    y_test = y_plot

    x_min = X_test[:, 0].min() - 1
    x_max = X_test[:, 0].max() + 1
    y_min = X_test[:, 1].min() - 1
    y_max = X_test[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=2, ncols=2, sharex="col",
                            sharey="row", figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):

        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)

        axarr[idx[0], idx[1]].scatter(
            X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="blue", marker="^", s=50
        )

        axarr[idx[0], idx[1]].scatter(
            X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="green", marker="o", s=50
        )

        axarr[idx[0], idx[1]].scatter(
            X_test[y_test == 2, 0], X_test[y_test == 2, 1], c="red", marker="x", s=50
        )

        axarr[idx[0], idx[1]].set_title(tt)
        axarr[idx[0], idx[1]].set_xlabel("Alcohol")
        axarr[idx[0], idx[1]].set_ylabel("Malic Acid")

    plt.tight_layout(pad=3.0)
    plt.show()
