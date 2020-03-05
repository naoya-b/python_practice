#畳み込みニューラルネットワーク 入力の形を保持して渡す。
x = np.random.rand(10,1,28,28)
x.shape # 10,1,28,28

x.[0] # (1,28,28)

#im2col
# coding: utf-8
import numpy as np
import collections


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]  #x[n:m:-1] x[n]からx[m]まで、-1ずつ進める。形は元のまま　 #np.r_（rバインド的な）
    w = np.kaiser(window_len, 2)  #https://numpy.org/doc/1.18/reference/generated/numpy.kaiser.html
    y = np.convolve(w/w.sum(), s, mode='valid') #畳み込み積分 https://deepage.net/features/numpy-convolve.html
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0]) #x.shape[0]=10 なら、0~9までの数をランダムに並べる。
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')  #https://qiita.com/horitaku1124/items/6ae979b21ddc7256b872
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


#convolution rayer
import sys,os
sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride = 1,pad = 0)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2,5,5,stride= 1 ,pad = 0)

class Convolution:
    def __init__(self,W,b,stride = 1,pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int(1+(H+2*self.pad - FH)/self.stride)
        out_w = int(1+(W+2*self.pad - FW)/self.stride)

        col = im2col(x,FH,FW,self.stride,self.pad)
        col_W = self.W.reshape(FN,1).T
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        return out


class Pooling:
    def __init__(self,pool_h,pool_w,stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1,self.pool_h*self.pool_w)

        out = np.max(col,axis = 1)
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        return out

# 7-5 CNN
class SimpleConvNet:
    def __init__(self,input_dim = (1,28,28),conv_param = {"filter_num":30,"filter_size":5,"pad":0,"stride":1},hidden_size = 100,output_size = 10,weight_init_std = 0.01):
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2* filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params["W1"] = weight_init_std*np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std*np.random.randn(pool_output_size,hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderedDict() #import collections https://note.nkmk.me/python-collections-ordereddict/
        self.layers["Conv1"] = Convolution(self.params["W1"],
                                           self.params["b1"],
                                           conv_param["stride"],
                                           conv_param["pad"])
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h = 2,pool_w = 2,stride = 2)
        self.layers["Affine1"] = Affine(self.params["W2"],self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"],self.params["b3"])

        self.last_layer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

   def loss(self,x,t):
       y = self.predict(x)
       return self.lastLayer.forward(y,t)

   def gradient(self,x,t):
       self.loss(x,t)
       dout = 1
       dout = self.lastLayer.backward(dout)

       layers = list(self.layers.values())
       layers.reverse()
       for layer in layers:
           dout = layer.backward(dout)

       grads = {}
       grads["W1"] = self.layers["Conv1"].dW
       grads["b1"] = self.layers["Conv1"].db
       grads["W2"] = self.layers["Affine1"].dW
       grads["b2"] = self.layers["Affine1"].db
       grads["W3"] = self.layers["Affine2"].dW
       grads["b3"] = self.layers["Affine2"].db

       return grads
