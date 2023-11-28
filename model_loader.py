class SingletonClass:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print('loading')
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._model = 'my model'
        return cls._instance

    def say_hello(self, text):
        print(f"model name : [{self._model}], input text [{text}]")
        return self._model