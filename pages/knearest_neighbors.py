import streamlit as st
from PIL import Image

st.title("K-Nearest Neighbors Algorithm")

header = '算法原理'
st.header(header)

msg = '''
k近邻(kNN)是一种监督学习算法，可用于解决分类和回归任务。\n
这个算法背后的主要思想是，一个数据点的值或类是由它周围的数据点决定的。\n
kNN分类器采用多数表决原则确定数据点的类别。由于模型需要存储所有的数据点，随着数据点数量的增加，kNN变得非常慢。\n
因此，它的内存效率也不高。kNN的另一个缺点是它对异常值很敏感。\n
'''
st.markdown(msg)

img = Image.open('images/K-Nearest Neighbors Algorithm.jpg')
st.image(img)

header = '示例代码'
st.header(header)

code = '''

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
 
irisData = load_iris()
 
# Create feature and target arrays
X = irisData.data
y = irisData.target
 
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
 
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
 
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
 
# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
'''

st.code(code)

if st.button("运行本例"):

# Import necessary modules
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    import numpy as np
    import matplotlib.pyplot as plt
    
    irisData = load_iris()
    
    # Create feature and target arrays
    X = irisData.data
    y = irisData.target
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.2, random_state=42)
    
    neighbors = np.arange(1, 9)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)
    
    # Generate plot
    fig = plt.figure()
    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
    
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    st.pyplot(fig)  