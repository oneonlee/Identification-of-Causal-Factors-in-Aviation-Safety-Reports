import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 필요한 라이브러리들을 가져옵니다.
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import numpy as np

# 예시 데이터 생성
data = np.random.randn(100, 4)  # 100개의 샘플과 4개의 특성을 가진 데이터
df = pd.DataFrame(data, columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"])

# PCA 수행
pca = PCA(n_components=2)  # 주성분 개수를 2로 설정
pca_result = pca.fit_transform(df)

# Streamlit 애플리케이션 시작
st.title("PCA 그래프")
st.write("PCA를 사용하여 데이터를 2차원으로 축소한 후 그래프로 표시합니다.")

# 그래프 그리기
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
st.pyplot()

# 데이터 테이블 출력
st.write("PCA를 적용한 데이터 테이블:")
st.dataframe(df)


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
fig = go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, mode="markers")])
fig.show()
