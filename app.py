import streamlit as st

# Using "with" notation

report_texts = []
file = open('./samples/sample1.txt', 'r')
text=file.read()
report_texts.append(text)
file.close()


with st.sidebar:
    reports = ['객실 화장실 내 흡연으로 인한 기내 화재','비행 중 군전투기와 충돌','이륙 후 새떼 충돌로 불시착']
    st.radio(
    "What's your favorite movie genre",
    [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
    captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."])
    st.radio(
    "What's your favorite movie genre",
    reports,
    captions = ["Laugh out loud.", "Get the popcorn.", "Never stop learning."])




st.title('항공 안전 사고 보고서 내 사고 원인 추출')

st.header('보고서')

txt = st.text_area(
    "보고서 본문",
    report_texts[0],height=300
    )

st.write(f'You wrote {len(txt)} characters.')

if st.button('analyze'):
    st.write(f'analyze : {txt}')
else:
    st.write('Goodbye')

st.header('사고 원인')

tab1, tab2 = st.tabs(["List", "Plot"])

with tab1:
    keywords=['one','two','three','four']
    idx=1
    for word in keywords:
        st.markdown(f'{idx}. {word}')
        idx+=1