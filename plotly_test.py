from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer


HUGGINGFACE_MODEL_PATH = 'jhgan/ko-sroberta-multitask'
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
  count = CountVectorizer(ngram_range=(4,6), lowercase=False, token_pattern=None, tokenizer=tokenizer).fit([doc])
  candidates = count.get_feature_names_out()
  return candidates


def plot_fig(keyphrases, doc):
  phrases = make_phrase2(doc)
  phrases = list(phrases)
  colors = []
  origin_phrases = []
  result = sbert_encode(phrases[0])
  origin_phrases.append(phrases[0])
  colors.append("green")
  phrases.pop(0)
  vector = None
  for phrase in phrases:
    vector = sbert_encode(phrase)
    result = np.append(result, vector, axis=0)
    colors.append("green")
    origin_phrases.append(phrase)

  for keyphrase in keyphrases:
    vector = sbert_encode(keyphrase)
    result = np.append(result, vector, axis=0)
    colors.append("blue")
    origin_phrases.append(keyphrase)

  df = pd.DataFrame(data=result, columns=range(0, vector.shape[1]))
  # 2차원 t-SNE 임베딩
  tsne_np = TSNE(n_components=2, perplexity=len(result) * 19 // 20).fit_transform(df)
  # numpy array -> DataFrame 변환
  tsne_df = pd.DataFrame(tsne_np, columns=['component 0', 'component 1'])
  tsne_df['phrase'] = origin_phrases
  fig = px.scatter(
      tsne_df, x='component 0', y='component 1', hover_data=['phrase'],hover_name='phrase',
      color=colors, labels={'color': 'number'}, color_discrete_map="identity"
  )
  return fig


def plot_fig_3d(keyphrases, doc):
  phrases = make_phrase2(doc)
  phrases = list(phrases)
  colors = []
  origin_phrases = []
  result = sbert_encode(phrases[0])
  origin_phrases.append(phrases[0])
  colors.append("green")
  phrases.pop(0)
  vector = None
  for phrase in phrases:
    vector = sbert_encode(phrase)
    result = np.append(result, vector, axis=0)
    colors.append("green")
    origin_phrases.append(phrase)

  for keyphrase in keyphrases:
    vector = sbert_encode(keyphrase)
    result = np.append(result, vector, axis=0)
    colors.append("blue")
    origin_phrases.append(keyphrase)

  df = pd.DataFrame(data=result, columns=range(0, vector.shape[1]))
  # 3차원 t-SNE 임베딩
  tsne_np_3 = TSNE(n_components=3, perplexity=len(result) * 19 // 20).fit_transform(df)
  # numpy array -> DataFrame 변환
  tsne_df_3 = pd.DataFrame(tsne_np_3, columns=['component 0', 'component 1', 'component 2'])
  tsne_df_3['phrase'] = origin_phrases
  fig = px.scatter_3d(tsne_df_3, x='component 0', y='component 1', z='component 2',
                      hover_data=['phrase'], hover_name='phrase', color=colors, color_discrete_map="identity")
  fig.update_traces(marker=dict(size=3))
  return fig
