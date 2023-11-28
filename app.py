import streamlit as st
import ai_function
import time
import model_loader

# Using "with" notation



# 모델 로더 객체 생성
report_analyzer_model = model_loader.SingletonClass()


report_texts = []
for i in range(1,5):
    file = open(f'./samples/sample{i}.txt', 'r')
    text=file.read()
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
    reports,
    captions = ["계기비행계획으로 마닐라를 출발, 서울을 향했다. FL370에서..",
                "G최근 B737 Rudder 문제로 이 기종에 대한 항공기 기동속도가..",
                "당일 기상조건은 밤이었으나 VMC로 적어도 7∼8Km 이상 시계가.."])



st.title('항공 안전 사고 보고서 내 사고 원인 추출')

st.header('보고서')
st.subheader(selected_option)
txt = st.text_area(label='본문',value=report_texts[reports_idxs[selected_option]],height=500,max_chars=850)

st.write(f'글자수: {len(txt)} 자')

if st.button('analyze'):
    phrases = txt.split()
    all_keyphrases=[]
    keyphrases=[]
    report_text_single_list=[report_texts[reports_idxs[selected_option]]]
    with st.spinner("Loading..."):
        all_keyphrases = report_analyzer_model.inference(test_report_list=report_text_single_list)
    keyphrases=all_keyphrases[reports_idxs[selected_option]]
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

tab1, tab2 = st.tabs(["Plot", "List"])

with tab1:
    keywords=['one','two','three','four']
    idx=1
    for word in keywords:
        st.markdown(f'{idx}. {word}')
        idx+=1