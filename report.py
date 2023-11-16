import streamlit as st

# Using "with" notation
with st.sidebar:
    reports = ['객실 화장실 내 흡연으로 인한 기내 화재','비행 중 군전투기와 충돌','이륙 후 새떼 충돌로 불시착']
    st.markdown("### reports")
    for report in reports:
        st.markdown(f"+ {report}")
    add_radio = st.radio(
        "Choose a shipping method",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )


st.title('항공 안전 사고 보고서 내 사고 원인 추출')

st.header('보고서')
title = st.text_input('보고서 본문', 'Life of Brian')
#st.write('The current movie title is', title)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.custom_value="hello_kth"

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable text input widget", key="disabled")
    st.radio(
        "Set text input label visibility 👉",
        key="visibility",
        options=["visible", "hidden", "collapsed"],
    )
    st.text_input(
        "Placeholder for the other text input widget",
        "This is a placeholder",
        key="placeholder",
    )

with col2:
    text_input = st.text_input(
        "Enter some text 👇",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder=st.session_state.placeholder,
    )

    if text_input:
        st.write("You entered: ", text_input)
        st.write("session state custom: ",st.session_state.custom_value)


txt = st.text_area(
    "Text to analyze",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair, (...)",height=200
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