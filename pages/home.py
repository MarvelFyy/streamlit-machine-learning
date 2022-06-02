import streamlit as st
from PIL import Image



st.title("Machine Learning")

image = Image.open('images/Machine Learning.jpg')

st.image(image, caption='机器学习常用算法分类')

msg = '''
机器学习是一门让计算机在没有明确编程的情况下行动的科学。在过去的十年里，机器学习为我们带来了自动驾驶汽车、实用的语音识别、有效的网络搜索，并极大地提高了对人类基因组的理解。机器学习在今天是如此的普遍，你可能每天使用它几十次而不知道它。许多研究人员还认为，这是向人类水平的人工智能取得进展的最佳方式。
'''
st.markdown(msg)