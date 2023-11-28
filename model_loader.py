import ai_function
import tensorflow as tf
from transformers import TFBertForTokenClassification, BertTokenizer


class SingletonClass:
    _instance = None
    _model = None
    _load_model = None
    _tokenizer = None
    MAX_SEQ_LEN = 850

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print('loading')
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._model = 'my model'
            cls.init_mode(cls)

        return cls._instance

    def inference(self,test_report_list, max_seq_len=MAX_SEQ_LEN):
        print(f"model name : [{self._model}], input text [{test_report_list}]")
        ai_function.inference(self._model, test_report_list, max_seq_len=max_seq_len, tokenizer=self._tokenizer)
        return self._model

    @staticmethod
    def init_mode(cls):
        HUGGINGFACE_MODEL_PATH = "oneonlee/KoAirBERT"
        cls._instance._tokenizer = BertTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)

        with tf.device(f"/GPU:0"):
            cls._instance._load_model = TFBertForTokenClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, num_labels=2, from_pt=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            cls._instance._load_model.compile(optimizer=optimizer, loss=cls._instance._load_model.hf_compute_loss)
            cls._instance._load_model.load_weights(f"{HUGGINGFACE_MODEL_PATH.replace('/', '-')}/tf_model.h5")