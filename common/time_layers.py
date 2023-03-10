# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid

class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self,x, h_prev):
        Wx, Wh, b = self.params
        t = np.dot(h_prev,Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)
        
        self.cache = (x, h_prev, h_next) #保存用
        return h_next

    def backward(self,dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)#tanhの微分
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful = False):#statefulは隠れ状態hを持つかどうか
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None,None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self,xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape #バッチサイズ、T個の時系列データ、入力ベクトルの次元
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H),dtype='f') #出力用の空の

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H),dtype='f') #初回呼び出しで0に初期化

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h) #statefuk=1の時は最後の層のメンバ変数hが残った状態
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx,Wh,b = self.params
        N, T, D = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N,T,D),dtype='f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)):# 時系列の後ろから回す
            layer = self.layers[t]
            dx,dh = layer.backward(dhs[:,t,:]+dh)
            dxs[:,t,:] = dx

            for i,grad in enumerate(layer.grads):
                grads[i] += grad #複数のRNNの勾配を足し合わせる

        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 入力`x`用の重みパラーメタ（4つ分の重みをまとめる）
        Wh: 隠れ状態`h`用の重みパラメータ（4つ分の重みをまとめる）
        b: バイアス（4つ分のバイアスをまとめる）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        #まとめて積を計算（計算量小）
        A = np.dot(x, Wx) + np.dot(h_prev,Wh) + b

        #ゲートごとに分ける
        f = A[:,:H]
        g = A[:,H:2*H]
        i = A[:,2*H:3*H]
        o = A[:,3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f* c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x,h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self,dh_next, dc_next):
        Wx,Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        #sigmoidの微分とtanhの微分
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        #連結
        dA = np.hstack((df,dg,di,do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self,Wx,Wh,b,stateful = False):#statefulは隠れ状態hを持つかどうか
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h,self.c,self.dh = None,None,None
        self.stateful = stateful

    def set_state(self, h, c=None):
        self.h = h
        self.c = c

    def reset_state(self):
        self.h, self.c = None, None

    def forward(self,xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape #バッチサイズ、T個の時系列データ、入力ベクトルの次元
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N,T,H),dtype='f') #出力用の空の

        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H),dtype='f') #初回呼び出しで0に初期化
        if not self.stateful or self.c is None:
            self.c = np.zeros((N,H),dtype='f') #初回呼び出しで0に初期化


        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:,t,:],self.h,self.c) #stateful=1の時は最後の層のメンバ変数hが残った状態
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx,Wh,b = self.params
        N, T, D = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N,T,D),dtype='f')
        dh,dc = 0, 0
        grads = [0,0,0]
        for t in reversed(range(T)):# 時系列の後ろから回す
            layer = self.layers[t]
            dx,dh,dc = layer.backward(dhs[:,t,:]+dh, dc)
            dxs[:,t,:] = dx

            for i,grad in enumerate(layer.grads):
                grads[i] += grad #複数のRNNの勾配を足し合わせる

        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs


class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask

