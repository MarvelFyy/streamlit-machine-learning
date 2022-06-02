import streamlit as st
from PIL import Image

st.title("K-Means Clustering Algorithm")

header = '算法原理'
st.header(header)

msg = '''
K均值聚类是最流行的聚类算法之一，是实践者在解决聚类任务时第一个应用的算法，以获得数据集的结构。\n
这是一种算法，给定一个数据集，将识别哪些数据点属于k个簇中的每一个，获取您的数据并学习如何对其进行分组。\n
通过一系列迭代，该算法创建了被称为集群的数据点组，这些数据点具有相似的方差，并最小化特定的代价函数:集群内的平方和。
'''
st.markdown(msg)

img = Image.open('images/K-Means Clustering Algorithm.jpg')
st.image(img)

header = '示例代码'
st.header(header)

code = '''
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

# Though the following import is not directly being used, it is required
# for 3D projection to work with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [
    ("k_means_iris_8", KMeans(n_clusters=8)),
    ("k_means_iris_3", KMeans(n_clusters=3)),
    ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
]

fignum = 1
titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
ax.set_position([0, 0, 0.95, 1])

for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
    ax.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2,
        name,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
ax.set_title("Ground Truth")
ax.dist = 12

fig.show()
'''

st.code(code)

if st.button('运行本例'):

    import numpy as np
    import matplotlib.pyplot as plt

    # Though the following import is not directly being used, it is required
    # for 3D projection to work with matplotlib < 3.2
    import mpl_toolkits.mplot3d  # noqa: F401

    from sklearn.cluster import KMeans
    from sklearn import datasets

    np.random.seed(5)

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    estimators = [
        ("k_means_iris_8", KMeans(n_clusters=8)),
        ("k_means_iris_3", KMeans(n_clusters=3)),
        ("k_means_iris_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
    ]

    fignum = 1
    titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]
    for name, est in estimators:
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
        ax.set_position([0, 0, 0.95, 1])
        est.fit(X)
        labels = est.labels_

        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor="k")

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel("Petal width")
        ax.set_ylabel("Sepal length")
        ax.set_zlabel("Petal length")
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.text3D(
            X[y == label, 3].mean(),
            X[y == label, 0].mean(),
            X[y == label, 2].mean() + 2,
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("Petal width")
    ax.set_ylabel("Sepal length")
    ax.set_zlabel("Petal length")
    ax.set_title("Ground Truth")
    ax.dist = 12

    st.pyplot(fig)