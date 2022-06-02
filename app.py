import streamlit as st
import streamlit_book as stb
from pathlib import Path

# Set multipage
current_path = Path(__file__).parent.absolute()

# Supervised Learning
stb.set_book_config(menu_title="机器学习算法",
                    menu_icon="",
                    options=[
                            "机器学习概述",
                            "线性回归",
                            "逻辑回归",
                            "决策树",
                            "随机森林",
                            "支持向量机",
                            "K最近邻算法",
                            "K均值聚类算法",
                            "层次聚类算法"
                            ],
                    paths=[
                        current_path / "pages/home.py",
                        current_path / "pages/linear_regression.py",
                        current_path / "pages/logistic_regression.py",
                        current_path / "pages/decision_tree.py",
                        current_path / "pages/random_forest.py",
                        current_path / "pages/support_vector_machine.py",
                        current_path / "pages/knearest_neighbors.py",
                        current_path / "pages/kmeans_clustering.py",
                        current_path / "pages/hierarchical_clustering.py"
                          ],
                    icons=[
                          "brightness-high-fill",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "trophy"
                          ],
                    save_answers=False,
                    )



with st.sidebar:

    st.sidebar.title("Author")
    st.sidebar.info(
        """
        @Nice小夫

        GitHub：https://github.com/MarvelFyy
        
        """
    )



