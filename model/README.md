# model

항공 안전 보고서에서 원인을 추출하기 위해 개발한 모델의 학습 및 추론 코드가 담겨 있는 디렉토리

1. BERT 계열 모델들을 Fine-tuning한 모델
    - KoAirBERT.ipynb
    - klue-bert-base.ipynb
    - klue-roberta-base.ipynb
2. KeyBERT 기반 시스템 (Backbone 모델을 변경하여 임베딩 추출)
    - KeyBERT-KLUE-BERT.ipynb
    - KeyBERT-KoAirBERT.ipynb
    - KeyBERT-SBERT.ipynb
3. SBERT를 전이학습한 모델
    - jhgan-ko-sroberta-multitask.ipynb
