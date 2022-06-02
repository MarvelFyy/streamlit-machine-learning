import streamlit as st
from PIL import Image

st.title("Logistic Regression")

header = '算法原理'
st.header(header)

msg = '''
逻辑回归是被称为分类算法的机器学习算法家族的一部分。它是解决二元分类问题的首选方法。\n
该算法通过拟合数据到logit函数来预测事件发生的概率。\n
因此，它也被称为logit回归。客户流失、垃圾邮件、网站或广告点击预测是逻辑回归提供强大解决方案的一些例子。
'''
st.markdown(msg)

img = Image.open('images/Logistic Regression.jpg')
st.image(img)

header = '示例代码'
st.header(header)

code = '''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

X, y = datasets.load_digits(return_X_y=True)

X = StandardScaler().fit_transform(X)

# classify small against large digits
y = (y > 4).astype(int)

l1_ratio = 0.5  # L1 weight in the Elastic-Net regularization

fig, axes = plt.subplots(3, 3)

# Set regularization parameter
for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
    # turn down tolerance for short training time
    clf_l1_LR = LogisticRegression(C=C, penalty="l1", tol=0.01, solver="saga")
    clf_l2_LR = LogisticRegression(C=C, penalty="l2", tol=0.01, solver="saga")
    clf_en_LR = LogisticRegression(
        C=C, penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, tol=0.01
    )
    clf_l1_LR.fit(X, y)
    clf_l2_LR.fit(X, y)
    clf_en_LR.fit(X, y)

    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()
    coef_en_LR = clf_en_LR.coef_.ravel()

    # coef_l1_LR contains zeros due to the
    # L1 sparsity inducing norm

    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
    sparsity_en_LR = np.mean(coef_en_LR == 0) * 100

    print("C=%.2f" % C)
    print("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_l1_LR))
    print("{:<40} {:.2f}%".format("Sparsity with Elastic-Net penalty:", sparsity_en_LR))
    print("{:<40} {:.2f}%".format("Sparsity with L2 penalty:", sparsity_l2_LR))
    print("{:<40} {:.2f}".format("Score with L1 penalty:", clf_l1_LR.score(X, y)))
    print(
        "{:<40} {:.2f}".format("Score with Elastic-Net penalty:", clf_en_LR.score(X, y))
    )
    print("{:<40} {:.2f}".format("Score with L2 penalty:", clf_l2_LR.score(X, y)))

    if i == 0:
        axes_row[0].set_title("L1 penalty")
        axes_row[1].set_title("Elastic-Net\nl1_ratio = %s" % l1_ratio)
        axes_row[2].set_title("L2 penalty")

    for ax, coefs in zip(axes_row, [coef_l1_LR, coef_en_LR, coef_l2_LR]):
        ax.imshow(
            np.abs(coefs.reshape(8, 8)),
            interpolation="nearest",
            cmap="binary",
            vmax=1,
            vmin=0,
        )
        ax.set_xticks(())
        ax.set_yticks(())

    axes_row[0].set_ylabel("C = %s" % C)

plt.show()
'''

st.code(code)

if st.button('运行本例'):
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    X, y = datasets.load_digits(return_X_y=True)

    X = StandardScaler().fit_transform(X)

    # classify small against large digits
    y = (y > 4).astype(int)

    l1_ratio = 0.5  # L1 weight in the Elastic-Net regularization

    fig, axes = plt.subplots(3, 3)

    # Set regularization parameter
    for i, (C, axes_row) in enumerate(zip((1, 0.1, 0.01), axes)):
        # turn down tolerance for short training time
        clf_l1_LR = LogisticRegression(C=C, penalty="l1", tol=0.01, solver="saga")
        clf_l2_LR = LogisticRegression(C=C, penalty="l2", tol=0.01, solver="saga")
        clf_en_LR = LogisticRegression(
            C=C, penalty="elasticnet", solver="saga", l1_ratio=l1_ratio, tol=0.01
        )
        clf_l1_LR.fit(X, y)
        clf_l2_LR.fit(X, y)
        clf_en_LR.fit(X, y)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        coef_l2_LR = clf_l2_LR.coef_.ravel()
        coef_en_LR = clf_en_LR.coef_.ravel()

        # coef_l1_LR contains zeros due to the
        # L1 sparsity inducing norm

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
        sparsity_en_LR = np.mean(coef_en_LR == 0) * 100

        st.write("C=%.2f" % C)
        st.write("{:<40} {:.2f}%".format("Sparsity with L1 penalty:", sparsity_l1_LR))
        st.write("{:<40} {:.2f}%".format("Sparsity with Elastic-Net penalty:", sparsity_en_LR))
        st.write("{:<40} {:.2f}%".format("Sparsity with L2 penalty:", sparsity_l2_LR))
        st.write("{:<40} {:.2f}".format("Score with L1 penalty:", clf_l1_LR.score(X, y)))
        st.write(
            "{:<40} {:.2f}".format("Score with Elastic-Net penalty:", clf_en_LR.score(X, y))
        )
        st.write("{:<40} {:.2f}".format("Score with L2 penalty:", clf_l2_LR.score(X, y)))

        if i == 0:
            axes_row[0].set_title("L1 penalty")
            axes_row[1].set_title("Elastic-Net\nl1_ratio = %s" % l1_ratio)
            axes_row[2].set_title("L2 penalty")

        for ax, coefs in zip(axes_row, [coef_l1_LR, coef_en_LR, coef_l2_LR]):
            ax.imshow(
                np.abs(coefs.reshape(8, 8)),
                interpolation="nearest",
                cmap="binary",
                vmax=1,
                vmin=0,
            )
            ax.set_xticks(())
            ax.set_yticks(())

        axes_row[0].set_ylabel("C = %s" % C)

    st.pyplot(fig)





