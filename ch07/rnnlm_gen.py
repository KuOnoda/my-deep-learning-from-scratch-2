import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skipids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x)
            p = softmax(score.flatten()) #予測値の確率分布（リスト）

            sampled = np.random.choice(len(p), size=1, p=p)
            if(skipids is None) or (sampled not in skipids):
                x = sampled
                word_ids.append(int(x))
        return word_ids 