import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    def __init__(self,vocab_size,hidden_size,window_size,corpus):
        V,H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(V,H).astype('f')

        self.in_layers = []
        for _ in range(2*window_size):#周囲どこまでをコンテキストとするか
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power = 0.75, sample_size=5)#負例のサンプル数5

        self.params, self.grads = [],[]
        layers = self.in_layers + [self.ns_loss]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        self.word_vecs = W_in #単語の分散表現を格納する

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,1])
        h*= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h,target)
        return loss

    def backward(self,dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1/len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
