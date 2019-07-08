import torch
import numpy as np

from model import get_model, load_model, load_vocab
from beamsearch import beam_search
from utils import seq2text

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
   
    def predict_current_word(self, inp, k=5, max_len=10, len_norm=False):
        seq = self.tokenizer.texts_to_sequences([inp])
        seq = torch.tensor(seq).long()
        end_token = self.tokenizer.word_index[' ']
        kseq, kscore = beam_search(seq, self.model, end_token, k, max_len)
        ksent = [seq2text(x, self.tokenizer, end_token) for x in kseq]

        kscore = np.exp(kscore)

        if len_norm:
            sent_len = [len(x) for x in ksent]
            kscore = kscore / sent_len
        
        res = {inp+sent:score for sent, score in zip(ksent, kscore)}
        return res

