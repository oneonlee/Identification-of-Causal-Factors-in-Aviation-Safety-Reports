import random

class SingletonClass:
    _instance = None
    _load_model = None
    _tokenizer = None
    MAX_SEQ_LEN = 850

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print('loading')
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls.init_model(cls)
        return cls._instance

    def inference(self,test_report_list, max_seq_len=MAX_SEQ_LEN):
        print(f"model name : [{self._load_model}], input text [{test_report_list}]")
        phrases = test_report_list[0].split()
        keyphrases = []
        indexes = random.sample(range(0,len(phrases)),5)
        for i in indexes:
            keyphrases.append(phrases[i])
        return keyphrases

    @staticmethod
    def init_model(cls):
        cls._instance._load_model = 'mock model'