from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


HUGGINGFACE_MODEL_PATH = 'jhgan/ko-sroberta-multitask'
GPU_NUM = 0
SEED = 42
SBERT_model = SentenceTransformer(HUGGINGFACE_MODEL_PATH)

def make_phrase(idx, phrase_list):
  res=phrase_list[idx]
  for i in range(1,5):
    if len(phrase_list)<=(idx+i):
      break
    res += f" {phrase_list[idx+i]}"
  #print(f"res : {res}")
  return res


def sample_plot_fig():
  file = open('./samples/demo1.txt', 'r')
  text = file.read()
  file.close()
  file = open('./samples/demo1_answer.txt', 'r')
  answer_text = file.read()
  print(text)
  print('answers')
  print(answer_text)

  phrases = text.split(' ')
  print(phrases)
  new_phrases = []
  for i in range(0, len(phrases), 5):
    new_phrases.append(make_phrase(i, phrases))
  phrases = new_phrases
  print('new phrases')
  print(phrases)
  keyphrases = answer_text.split(', ')
  print(keyphrases)

  colors = []
  origin_phrases = []
  result = SBERT_model.encode([phrases[0]])
  origin_phrases.append(phrases[0])
  colors.append("green")
  phrases.pop(0)
  vector=None
  for phrase in phrases:
    vector = SBERT_model.encode([phrase])
    result = np.append(result, vector, axis=0)
    colors.append("green")
    origin_phrases.append(phrase)

  for keyphrase in keyphrases:
    vector = SBERT_model.encode([keyphrase])
    result = np.append(result,vector,axis=0)
    colors.append("blue")
    origin_phrases.append(keyphrase)

  df = pd.DataFrame(data=result, columns=range(0, vector.shape[1]))

  # 2차원 t-SNE 임베딩
  tsne_np = TSNE(n_components = 2,perplexity=len(result)*19//20).fit_transform(df)

  # numpy array -> DataFrame 변환
  tsne_df = pd.DataFrame(tsne_np, columns=['component 0', 'component 1'])
  tsne_df['phrase']=origin_phrases

  fig = px.scatter(
      tsne_df, x='component 0', y='component 1', hover_data=['phrase'],hover_name='phrase',
      color=colors, labels={'color': 'number'}, color_discrete_map="identity"
  )
  return fig


def sample_plot_fig_3d():
  file = open('./samples/demo1.txt', 'r')
  text = file.read()
  file.close()
  file = open('./samples/demo1_answer.txt', 'r')
  answer_text = file.read()
  print(text)
  print('answers')
  print(answer_text)

  phrases = text.split(' ')
  print(phrases)
  new_phrases = []
  for i in range(0, len(phrases), 5):
    new_phrases.append(make_phrase(i, phrases))
  phrases = new_phrases
  print('new phrases')
  print(phrases)
  keyphrases = answer_text.split(', ')
  print(keyphrases)

  colors = []
  origin_phrases = []
  result = SBERT_model.encode([phrases[0]])
  origin_phrases.append(phrases[0])
  colors.append("green")
  phrases.pop(0)
  vector=None
  for phrase in phrases:
    vector = SBERT_model.encode([phrase])
    result = np.append(result, vector, axis=0)
    colors.append("green")
    origin_phrases.append(phrase)

  for keyphrase in keyphrases:
    vector = SBERT_model.encode([keyphrase])
    result = np.append(result,vector,axis=0)
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
  return fig
