## 데모 실행 방법

- 초기 설정
```shell
conda create -n air python=3.9
cd C:\Users\dilabpc\Project
git clone https://github.com/johan1103/air_ice.git
cd air_ice
conda activate air
pip install -r requirements.txt
wget https://drive.google.com/u/0/uc?id=1c-sPiKEz--hAhkhwxBjv1daL5Jc8pj0H&export=download&confirm=t&uuid=7c3b85d0-08eb-4a1e-afa1-c964b152d6f7&at=AB6BwCDH4cA8hXII4tIrQeNJVEi2:1701181226490 -P oneonlee-KoAirBERT/
```

- 데모 서버 호스팅
```shell
conda activate air
streamlit run app.py
```
