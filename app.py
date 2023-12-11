import streamlit as st
import model_loader
import plotly_test

st.set_page_config(
    page_title="항공 안전 보고서 내 원인 요인 식별",
    page_icon="✈️",
    # layout="wide",
    initial_sidebar_state="expanded",
)


# 모델 로더 객체 생성
report_analyzer_model = model_loader.SingletonClass()


report_texts = []
for i in range(0, 8):
    file = open(f"./samples/sample{i}.txt", "r", encoding="UTF-8")
    text = file.read()
    report_texts.append(text)
    file.close()


reports_idxs = {}
selected_option = 0
with st.sidebar:
    reports = [
        "비행 중 기내난동",
        "지상 활주 중 기내 환자 발생으로 Ramp Return",
        "지상 기내식 직원의 표준운영절차",
        "운항 중 실수로 인한 일시적 통신두절",
        "이륙 중 타이어 파열",
        "항공기 운항 중 기내 환자 사망",
        "항공기 납치 협박 사건",
        "수하물 낙하 및 환자 이송으로 인한 지연",
    ]
    r_idx = 0
    for r in reports:
        reports_idxs[r] = r_idx
        r_idx += 1
    selected_option = st.radio("보고서 목록", reports)


st.title("항공 안전 보고서 내 원인 요인 식별")
st.header(f"<{selected_option}>")
txt = st.text_area(
    label="본문",
    value=report_texts[reports_idxs[selected_option]],
    height=500,
    max_chars=850,
)

cluster_size = st.slider(
    "Select Cluster Size",
    value=5, min_value=3, max_value=10)
st.write("cluster size : ", cluster_size)

keyphrases = []
if st.button("Start Analyze 🛫"):
    all_keyphrases = []
    report_text_single_list = [txt]
    with st.spinner("Loading..."):
        all_keyphrases = report_analyzer_model.inference(
            test_report_list=report_text_single_list
        )
    keyphrases = all_keyphrases[0]
    print("keyphrases: ", keyphrases)

    splitted_keyphrases = [[kp.split(" ")] for kp in keyphrases]

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
st.markdown(body="<br><br>", unsafe_allow_html=True)

st.header("사고 원인")

tab1, tab2 = st.tabs(["3D t-SNE", "2D t-SNE"])

with tab1:
    with st.spinner("Loading..."):
        graph = plotly_test.plot_fig_3d(keyphrases, txt, cluster_size=cluster_size)
        if graph is not None:
            st.plotly_chart(plotly_test.plot_fig_3d(keyphrases, txt, cluster_size=cluster_size),
                            use_container_width=True, theme=None)
with tab2:
    with st.spinner("Loading..."):
        graph = plotly_test.plot_fig(keyphrases, txt, cluster_size=cluster_size)
        if graph is not None:
            st.plotly_chart(plotly_test.plot_fig(keyphrases, txt, cluster_size=cluster_size),
                            use_container_width=True, theme=None)
