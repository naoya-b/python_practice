import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict  #https://qiita.com/apollo_program/items/165fb01b52702274936c

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(hidden_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"],self.params["b1"]) #コンストラクタに入れている。
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"],self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        for layer in self.layer.values(): #https://www.javadrive.jp/python/dictionary/index8.html
            x = layer.forward(x)

        return x
    #x:入力データ　t 教師データ
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1)
        if t.ndim != 1 : t = np.argmax(t,axis = 1)

        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)

        grads = {} #すでに一度loss_Wには初期データが入っている。
        grads["W1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.params["b2"])

        return grads
    def gradient(self,x,t):
        #forward
        self.loss(x,t)
        #backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
