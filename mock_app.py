import streamlit as st
import time
import mock_model_loader
import plotly_test

# Using "with" notation



# 모델 로더 객체 생성
report_analyzer_model = mock_model_loader.SingletonClass()


report_texts = []
for i in range(1,5):
    file = open(f'./samples/sample{i}.txt', 'r',encoding='UTF-8')
    text = file.read()
    report_texts.append(text)
    file.close()


reports_idxs={}
selected_option=0
with st.sidebar:
    reports = ['객실 화장실 내 흡연으로 인한 기내 화재','비행 중 군전투기와 충돌','이륙 후 새떼 충돌로 불시착']
    r_idx=0
    for r in reports:
        reports_idxs[r]=r_idx
        r_idx+=1
    selected_option=st.radio(
    "보고서 목록",
    reports)



st.title('항공 안전 사고 보고서 내 사고 원인 추출')

st.header('보고서')
st.subheader(selected_option)
txt = st.text_area(label='본문',value=report_texts[reports_idxs[selected_option]], height=250, max_chars=850)

st.write(f'글자수: {len(txt)} 자')
cluster_size = st.slider(
    "Select Cluster Size",
    value=5, min_value=3, max_value=10)

keyphrases = []
if st.button('Start Analyze 🛫'):
    phrases = txt.split()
    all_keyphrases = []
    report_text_single_list = [txt]
    print(f"report text single list type : {type(report_text_single_list)}")
    with st.spinner("Loading..."):
        all_keyphrases = report_analyzer_model.inference(
            test_report_list=report_text_single_list
        )
    keyphrases = all_keyphrases[0]

    analyze_result = "<div>" + txt
    for key in keyphrases:
        target = key
        insertion_front = '<span style="background-color: yellow;">'
        insertion_back = "</span>"
        index = analyze_result.find(target)

        if index != analyze_result.rfind(target):
            continue
        if index != -1:
            analyze_result = (
                analyze_result[:index]
                + insertion_front
                + target
                + insertion_back
                + analyze_result[index + len(target) :]
            )
    analyze_result += "</div>"
    st.markdown(body=analyze_result, unsafe_allow_html=True)
st.markdown(body='<br><br>',unsafe_allow_html=True)

st.header('사고 원인')

tab1, tab2 = st.tabs(["Plot_2d", "Plot_3d"])
with tab1:
    with st.spinner("Loading..."):
        graph = plotly_test.plot_fig(keyphrases, txt, cluster_size=cluster_size)
        if graph is not None:
            st.plotly_chart(plotly_test.plot_fig(keyphrases, txt, cluster_size=cluster_size
                                                 ), use_container_width=True, theme=None)

with tab2:
    with st.spinner("Loading..."):
        graph = plotly_test.plot_fig_3d(keyphrases, txt)
        if graph is not None:
            st.plotly_chart(plotly_test.plot_fig_3d(keyphrases, txt, cluster_size=cluster_size),
                            use_container_width=True, theme=None)
