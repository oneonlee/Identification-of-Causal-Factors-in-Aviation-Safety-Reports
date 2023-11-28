import streamlit as st
import ai_function
import time
import model_loader

# Using "with" notation



# 모델 로더 객체 생성
report_analyzer_model = model_loader.SingletonClass()


report_texts = []
for i in range(1,5):
    file = open(f'./samples/sample{i}.txt', 'r',encoding='UTF-8')
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
    reports)



st.title('항공 안전 사고 보고서 내 사고 원인 추출')

st.header('보고서')
st.subheader(selected_option)
txt = st.text_area(label='본문',value=report_texts[reports_idxs[selected_option]],height=500,max_chars=850)

st.write(f'글자수: {len(txt)} 자')

if st.button('analyze'):
    phrases = txt.split(" ")
    all_keyphrases=[]
    keyphrases=[]
    report_text_single_list=[report_texts[reports_idxs[selected_option]]]
    with st.spinner("Loading..."):
        all_keyphrases = report_analyzer_model.inference(test_report_list=report_text_single_list)
    keyphrases=all_keyphrases[0]
    splitted_keyphrases=[[kp.split(" ")] for kp in keyphrases]
    
    analyze_result="<div>"+txt
    
    for key in keyphrases:
        target=key
        insertion_front='<span style="background-color: yellow;">'
        insertion_back='</span>'
        index=analyze_result.find(target)
        if index!=-1:
            analyze_result=analyze_result[:index]+insertion_front+target+insertion_back+analyze_result[index+len(target):]
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