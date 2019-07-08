import torch

from model import get_model, load_model, load_vocab

class Singleton(type):
    """
    Define an Instance operation that lets clients access its unique
    instance.
    """

    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Predictor(metaclass=Singleton):
    def init_model(self, model_path, vocab_path):
        self.model = get_model()
        load_model(self.model, model_path)
        self.model.eval()
        self.tokenizer = load_vocab(vocab_path)


    def predict_next_char(self, inp, k=5):
        with torch.no_grad():
            seq = self.tokenizer.texts_to_sequences([inp])
            seq = torch.tensor(seq)
            prob, hs = self.model.predict(seq)
            kprob, kix = prob[0,-1,:].topk(k)
            res = {self.tokenizer.index_word[i]:float(p) for p,i in zip(kprob.detach().numpy(), kix.detach().numpy())}
            # res = {k:res[k] for k in sorted(res, key=res.get, reverse=True)}
            return res
   
