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
txt = st.text_area(label='본문',value=report_texts[reports_idxs[selected_option]],height=500,max_chars=850)

st.write(f'글자수: {len(txt)} 자')

if st.button('analyze'):
    phrases = txt.split()
    all_keyphrases = []
    keyphrases = []
    report_text_single_list = [txt]
    print(f"report text single list type : {type(report_text_single_list)}")
    with st.spinner("Loading..."):
        all_keyphrases = report_analyzer_model.inference(test_report_list=report_text_single_list)
    keyphrases = all_keyphrases[reports_idxs[selected_option]]
    keyDict = {}
    analyze_result='<div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px;">'
    for k in keyphrases:
        keyDict[k]=k
    for p in phrases:
        if p in keyDict:
            analyze_result+=f'<span style="background-color: yellow;">{p}</span>'
            #analyze_result+=f'**{p}** '
        else:
            analyze_result+=f'{p} '
    analyze_result+="</div>"
    st.markdown(body=analyze_result,unsafe_allow_html=True)
st.markdown(body='<br><br>',unsafe_allow_html=True)

st.header('사고 원인')

tab1, tab2, tab3 = st.tabs(["List", "Plot_2d", "Plot_3d"])

with tab1:
    keywords=['one', 'two', 'three', 'four']
    idx=1
    for word in keywords:
        st.markdown(f'{idx}. {word}')
        idx+=1

with tab2:
    st.plotly_chart(plotly_test.sample_plot_fig(), use_container_width=True, theme=None)

with tab3:
    st.plotly_chart(plotly_test.sample_plot_fig_3d(), use_container_width=True, theme=None)
