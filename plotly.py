import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 필요한 라이브러리들을 가져옵니다.
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np


# 가정: 여기서 'high_dimensional_data'는 당신이 가진 고차원 데이터입니다.
# 데이터를 랜덤하게 생성하겠습니다.
high_dimensional_data = np.random.rand(100, 10) 

# t-SNE 모델을 만들고 데이터를 3차원으로 변환합니다.
model = TSNE(n_components=3)
reduced_data = model.fit_transform(high_dimensional_data)

# 변환된 데이터를 x, y, z 좌표로 분리합니다.
xs = reduced_data[:, 0]
ys = reduced_data[:, 1]
zs = reduced_data[:, 2]

# Plotly를 사용해 3D 산점도를 만듭니다.
fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers')])

st.plotly_chart(fig)
