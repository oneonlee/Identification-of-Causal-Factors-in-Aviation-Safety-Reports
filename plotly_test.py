from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from plotly.validators.scatter.marker import SymbolValidator

HUGGINGFACE_MODEL_PATH = "jhgan/ko-sroberta-multitask"
GPU_NUM = 0
SEED = 42
SBERT_model = SentenceTransformer(HUGGINGFACE_MODEL_PATH)

vector_memory = {}


def sbert_encode(string):
    if string in vector_memory:
        return vector_memory[string]
    vector = SBERT_model.encode([string])
    vector_memory[string] = vector
    return vector


def tokenizer(text):
    return text.split(" ")


def make_phrase2(doc, phrase_list=[]):
    count = CountVectorizer(
        ngram_range=(4, 6), lowercase=False, token_pattern=None, tokenizer=tokenizer
    ).fit([doc])
    candidates = count.get_feature_names_out()
    return candidates


def plot_fig(keyphrases, doc, cluster_size=5):
    if len(keyphrases) == 0:
        return None

    phrases = make_phrase2(doc)
    phrases = list(phrases)
    origin_phrases = []
    plot_size = []
    result = sbert_encode(phrases[0])
    origin_phrases.append(phrases[0])
    phrases.pop(0)
    plot_size.append(1)
    vector = None
    for phrase in phrases:
        vector = sbert_encode(phrase)
        result = np.append(result, vector, axis=0)
        origin_phrases.append(phrase)
        plot_size.append(1)

    keyphrases_dict = {}
    for keyphrase in keyphrases:
        vector = sbert_encode(keyphrase)
        result = np.append(result, vector, axis=0)
        origin_phrases.append(keyphrase)
        keyphrases_dict[keyphrase] = True
        plot_size.append(3)

    df = pd.DataFrame(data=result, columns=range(0, vector.shape[1]))
    # 2차원 t-SNE 임베딩
    tsne_np = TSNE(n_components=2, perplexity=len(result) * 19 // 20).fit_transform(df)
    # numpy array -> DataFrame 변환
    tsne_df = pd.DataFrame(tsne_np, columns=["component 0", "component 1"])
    tsne_df["phrase"] = origin_phrases

    key_clusters = {}
    cluster_sample_df = tsne_df[["component 0", "component 1"]]
    print(f"cluster size : {cluster_size}")
    kmeans = KMeans(n_clusters=cluster_size)
    kmeans.fit(cluster_sample_df)
    result_by_sklearn = cluster_sample_df.copy()
    result_by_sklearn["cluster"] = kmeans.labels_
    cluster_sample_df["phrase"] = origin_phrases
    clusters = list(result_by_sklearn["cluster"])
    cluster_phrase = list(cluster_sample_df["phrase"])

    symbols = []
    colors = []
    symbol_sequence = ["circle-open", "circle", "square"]
    color_converter = [
        "red",
        "green",
        "orange",
        "blue",
        "purple",
        "black",
        "yellow",
        "goldenrod",
        "magenta",
        "pink",
        "brown",
    ]
    for idx, c in enumerate(clusters):
        sen = cluster_phrase[idx]
        # symbols.append(shape_converter[c])
        colors.append(color_converter[c])
        if sen in keyphrases_dict:
            key_clusters[c] = True
    for idx, c in enumerate(clusters):
        if c in key_clusters:
            if cluster_phrase[idx] in keyphrases_dict:
                symbols.append(1)
            else:
                symbols.append(2)
            # colors.append('black')
        else:
            symbols.append(3)
            # colors.append(color_converter[c])

    cluster_sample_df["symbol"] = symbols
    cluster_sample_df["color"] = colors

    print(cluster_sample_df)
    fig = px.scatter(
        cluster_sample_df,
        x="component 0",
        y="component 1",
        hover_data="phrase",
        hover_name="phrase",
        color="color",
        color_discrete_map="identity",
        symbol="symbol",
        size=plot_size,
        symbol_map={1: "square", 2: "square-open", 3: "circle-open"},
        opacity=1,
    )
    # fig.update_traces(mode='markers', marker=dict(size=10))
    fig.update_layout(showlegend=False)
    # fig.update_traces(opacity=1)
    return fig


def plot_fig_3d(keyphrases, doc, cluster_size=5):
    if len(keyphrases) == 0:
        return None
    phrases = make_phrase2(doc)
    phrases = list(phrases)
    origin_phrases = []
    plot_size = []
    result = sbert_encode(phrases[0])
    origin_phrases.append(phrases[0])
    phrases.pop(0)
    plot_size.append(3)
    vector = None
    for phrase in phrases:
        vector = sbert_encode(phrase)
        result = np.append(result, vector, axis=0)
        origin_phrases.append(phrase)
        plot_size.append(3)

    keyphrases_dict = {}
    for keyphrase in keyphrases:
        vector = sbert_encode(keyphrase)
        result = np.append(result, vector, axis=0)
        origin_phrases.append(keyphrase)
        keyphrases_dict[keyphrase] = True
        plot_size.append(15)

    df = pd.DataFrame(data=result, columns=range(0, vector.shape[1]))
    # 3차원 t-SNE 임베딩
    tsne_np_3 = TSNE(n_components=3, perplexity=len(result) * 19 // 20).fit_transform(
        df
    )
    # numpy array -> DataFrame 변환
    tsne_df_3 = pd.DataFrame(
        tsne_np_3, columns=["component 0", "component 1", "component 2"]
    )
    tsne_df_3["phrase"] = origin_phrases

    key_clusters = {}
    cluster_sample_df = tsne_df_3[["component 0", "component 1", "component 2"]]
    kmeans = KMeans(n_clusters=cluster_size)
    kmeans.fit(cluster_sample_df)
    result_by_sklearn = cluster_sample_df.copy()
    result_by_sklearn["cluster"] = kmeans.labels_
    cluster_sample_df["phrase"] = origin_phrases
    clusters = list(result_by_sklearn["cluster"])
    cluster_phrase = list(cluster_sample_df["phrase"])

    symbols = []
    colors = []
    shape_converter = [
        "star",
        "square",
        "diamond",
        "cross",
        "x",
        "hourglass",
        "hexagona",
    ]
    color_converter = [
        "red",
        "green",
        "orange",
        "blue",
        "purple",
        "black",
        "yellow",
        "goldenrod",
        "magenta",
        "pink",
        "brown",
    ]

    for idx, c in enumerate(clusters):
        sen = cluster_phrase[idx]
        # symbols.append(shape_converter[c])
        colors.append(color_converter[c])
        if sen in keyphrases_dict:
            key_clusters[c] = True
    for idx, c in enumerate(clusters):
        if c in key_clusters:
            if cluster_phrase[idx] in keyphrases_dict:
                symbols.append(1)
            else:
                symbols.append(2)
            # colors.append('black')
        else:
            symbols.append(3)
            # colors.append(color_converter[c])

    cluster_sample_df["symbol"] = symbols
    cluster_sample_df["color"] = colors

    fig = px.scatter_3d(
        cluster_sample_df,
        x="component 0",
        y="component 1",
        z="component 2",
        hover_data=["phrase"],
        hover_name="phrase",
        color="color",
        color_discrete_map="identity",
        symbol=symbols,
        size=plot_size,
        symbol_map={1: "square", 2: "square-open", 3: "circle-open"},
        opacity=1,
    )
    """
  fig.update_traces(
    marker=dict(
      size=[10 if c == 'red' else 5 for c in cluster_sample_df['color']]
    )
  )
  """
    fig.update_layout(showlegend=False)
    # fig.update_traces(marker=dict(size=3))
    return fig
