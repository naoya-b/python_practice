#TwoLayerNet
import sys, os,pickle
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
     return 1/(1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a)
    sum_a = np.sum(exp_a) # これは、aが大きいと厳しい。
    y = exp_a/sum_a
    return y

def cross_entropy_error(y,t): #最小二乗誤差
    t = np.array(t).reshape(1,t.size).T
    delta = 1e-7#10^-7
    return -np.sum(t*np.log(y + delta)) #deltaを入れることで、np.log(0) = infを防ぐ.

def numerical_gradient(f,x): #fのxでの偏微分
    x = x.astype(float)
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x.flatten()[idx]
        x.flatten()[idx] = tmp_val + h
        fxh1 = f(x) #f(x+h)

        x.flatten()[idx] = tmp_val - h
        fxh2 = f(x) #f(x-h)
        #以上二つで、ある次元のxのみ中心差分を取ってる。

        grad.flatten()[idx] = (fxh1 - fxh2) / (2*h)
        x.flatten()[idx] = tmp_val
    return grad


class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size,hidden_size) #random_randnは(2,3)なら2*3この乱数を作る
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self,x):
        W1,W2 = self.params["W1"],self.params["W2"]
        b1,b2 = self.params["b1"],self.params["b2"]
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y

    def loss (self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y= np.argmax(y,axis = 1)
        t = np.argmax(t,axis = 1)

        accuracy= np.sum(y == t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self,x,t):
        def loss_W(W):
            return self.loss(x,t)
        grads = {}
        grads["W1"] = numerical_gradient(loss_W,self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W,self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W,self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W,self.params["b2"])

        return grads

#ミニバッチ学習
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
train_loss_list = []

#ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = int(input("batch_size="))
learning_rate = 0.1

network = TwoLayerNet(input_size = x_train.shape[1],hidden_size = 50,output_size = 10)
for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size) #trainsize = 60000から、100個をランダムで選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.numerical_gradient(x_batch,t_batch) #微分する。
    #パラメータの更新
    for key in ("W1","b1","W2","b2"):
        network.params[key] -=  learning_rate*grad[key]
    #学習経過
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
