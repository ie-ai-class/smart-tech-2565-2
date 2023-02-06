import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def plot_decision_surface_train_test(
    X_train, X_test, y_train, y_test, clf, filename=""
):

    if (
        (type(X_train) != np.ndarray)
        | (type(X_test) != np.ndarray)
        | (type(y_train) != np.ndarray)
        | (type(y_test) != np.ndarray)
    ):
        print("\n\nPlotting Error:X and Y needs to be NumPy arrays.")
        return
    if (len(X_train.shape) != 2) | (len(X_test.shape) != 2):
        print("\n\nPlotting Error: X needs to be a 2D NumPy array.")
    if (X_train.shape[1] != 2) | (X_test.shape[1] != 2):
        print("\n\nPlotting Error: X needs to have 2 columns.")
        return

    if isinstance(clf, LogisticRegression):
        fitType = "logistic"
    elif isinstance(clf, Perceptron):
        fitType = "perceptron"
    elif isinstance(clf, SVC):
        fitType = "svm"
    elif isinstance(clf, DecisionTreeClassifier):
        fitType = "tree"
    elif isinstance(clf, RandomForestClassifier):
        fitType = "forest"
    elif isinstance(clf, KNeighborsClassifier):
        fitType = "knn"
    else:
        print("\n\nUnknown classifier: " + type(clf))
        return

    # setup
    resolution = 0.02
    markers = ("s", "x", "o", "^", "v")
    linestyles = (":", "--", "-.")

    X = X_train
    y = y_train
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    if fitType == "perceptron":
        # Fig 1
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # Plot hyperplane
        for idx, (W, w0) in enumerate(zip(clf.coef_, clf.intercept_)):
            hm = -W[0] / W[1]
            hx = np.linspace(x1_min, x1_max)
            hy = hm * hx - (w0) / W[1]
            ax1.plot(hx, hy, linestyle=linestyles[idx])
        # Plot data
        for idx, cl in enumerate(np.unique(y)):
            ax1.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.6,
                # edgecolor="black",
                cmap="Set3",
                marker=markers[idx],
                label=cl,
            )
        ax1.set_title("Training Data")
        ax1.set_xlim([xx1.min(), xx1.max()])
        ax1.set_ylim([xx2.min(), xx2.max()])
        ax1.set_xlabel("petal length (scaled)")
        ax1.set_ylabel("petal width (scaled)")
        ax1.legend()

        X = X_test
        y = y_test
        # Plot hyperplane
        for idx, (W, w0) in enumerate(zip(clf.coef_, clf.intercept_)):
            hm = -W[0] / W[1]
            hx = np.linspace(x1_min, x1_max)
            hy = hm * hx - (w0) / W[1]
            ax2.plot(hx, hy, linestyle=linestyles[idx])
        # Plot data
        for idx, cl in enumerate(np.unique(y)):
            ax2.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.6,
                # edgecolor="black",
                cmap="Set3",
                marker=markers[idx],
                label=cl,
            )
        ax2.set_title("Testing Data")
        ax2.set_xlim([xx1.min(), xx1.max()])
        ax2.set_ylim([xx2.min(), xx2.max()])
        ax2.set_xlabel("petal length (scaled)")
        ax2.set_ylabel("petal width (scaled)")
        ax2.legend()

    # Fig 2
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))

    X = X_train
    y = y_train
    # Contour plot
    ax3.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3")
    # Plot data
    for idx, cl in enumerate(np.unique(y)):
        ax3.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            # edgecolor="black",
            cmap="Set3",
            marker=markers[idx],
            label=cl,
        )
    ax3.set_title("Training Data")
    ax3.set_xlim([xx1.min(), xx1.max()])
    ax3.set_ylim([xx2.min(), xx2.max()])
    ax3.set_xlabel("petal length (scaled)")
    ax3.set_ylabel("petal width (scaled)")
    ax3.legend()

    X = X_test
    y = y_test
    # Contour plot
    ax4.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3")
    # Plot data
    for idx, cl in enumerate(np.unique(y)):
        ax4.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            # edgecolor="black",
            cmap="Set3",
            marker=markers[idx],
            label=cl,
        )
    ax4.set_title("Testing Data")
    ax4.set_xlim([xx1.min(), xx1.max()])
    ax4.set_ylim([xx2.min(), xx2.max()])
    ax4.set_xlabel("petal length (scaled)")
    ax4.set_ylabel("petal width (scaled)")
    ax4.legend()

    if filename:
        loc = filename.find(".")
        fname = filename[:loc]
        ext = filename[loc:]
        if fitType == "perceptron":
            fig1.savefig("./" + fname + "_1" + ext, dpi=300)
        fig2.savefig("./" + fname + "_2" + ext, dpi=300)


def plot_decision_surface_predict(X_train, y_train, clf, filename=""):

    if (type(X_train) != np.ndarray) | (type(y_train) != np.ndarray):
        print("\n\nPlotting Error:X and Y needs to be NumPy arrays.")
        return
    if len(X_train.shape) != 2:
        print("\n\nPlotting Error: X needs to be a 2D NumPy array.")
    if X_train.shape[1] != 2:
        print("\n\nPlotting Error: X needs to have 2 columns.")
        return

    if isinstance(clf, LogisticRegression):
        fitType = "logistic"
    elif isinstance(clf, Perceptron):
        fitType = "perceptron"
    elif isinstance(clf, SVC):
        fitType = "svm"
    elif isinstance(clf, DecisionTreeClassifier):
        fitType = "tree"
    elif isinstance(clf, RandomForestClassifier):
        fitType = "forest"
    elif isinstance(clf, KNeighborsClassifier):
        fitType = "knn"
    else:
        print("\n\nUnknown classifier: " + type(clf))
        return

    # setup
    resolution = 0.02
    markers = ("s", "x", "o", "^", "v")
    linestyles = (":", "--", "-.")

    X = X_train
    y = y_train
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    if fitType == "perceptron":
        # Fig 1
        fig1, (ax1) = plt.subplots(1, 1, figsize=(5, 5))
        # Plot hyperplane
        for idx, (W, w0) in enumerate(zip(clf.coef_, clf.intercept_)):
            hm = -W[0] / W[1]
            hx = np.linspace(x1_min, x1_max)
            hy = hm * hx - (w0) / W[1]
            ax1.plot(hx, hy, linestyle=linestyles[idx])
        # Plot data
        for idx, cl in enumerate(np.unique(y)):
            ax1.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.6,
                # edgecolor="black",
                cmap="Set3",
                marker=markers[idx],
                label=cl,
            )
        ax1.set_title("Unlabeled Data")
        ax1.set_xlim([xx1.min(), xx1.max()])
        ax1.set_ylim([xx2.min(), xx2.max()])
        ax1.set_xlabel("petal length (scaled)")
        ax1.set_ylabel("petal width (scaled)")
        ax1.legend()

    # Fig 2
    fig2, (ax3) = plt.subplots(1, 1, figsize=(5, 5))

    X = X_train
    y = y_train
    # Contour plot
    ax3.contourf(xx1, xx2, Z, alpha=0.4, cmap="Set3")
    # Plot data
    for idx, cl in enumerate(np.unique(y)):
        ax3.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            # edgecolor="black",
            cmap="Set3",
            marker=markers[idx],
            label=cl,
        )
    ax3.set_title("Unlabeled Data")
    ax3.set_xlim([xx1.min(), xx1.max()])
    ax3.set_ylim([xx2.min(), xx2.max()])
    ax3.set_xlabel("petal length (scaled)")
    ax3.set_ylabel("petal width (scaled)")
    ax3.legend()

    if filename:
        loc = filename.find(".")
        fname = filename[:loc]
        ext = filename[loc:]
        if fitType == "perceptron":
            fig1.savefig("./" + fname + "_1" + ext, dpi=300)
        fig2.savefig("./" + fname + "_2" + ext, dpi=300)
