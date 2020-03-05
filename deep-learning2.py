#事前準備 + 確認は終わり
#cd cd01
import sys, os,pickle
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)

#ニューラルネットワークの推論処理
x_train[0].shape #(784,)つまり画面サイズ28*28 = 784個のデータが入っている。そこから一つの数字が割り当てラれてる
x_train.shape #(60000,784) つまり、60000個数字がある

#ニューロン層の作成
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test,t_test
def init_network():
    with open ("sample_weight.pkl","rb") as f: #自動的にopenしたファイルをcloseしてくれる。また、fという名前をつけて動かせる。
        network = pickle.load(f)
    return network
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0

for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print((accuracy_cnt)/len(x))

#バッチ:n枚の画像を一気に処理したい。バッチとは束のこと
batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size): #batch_sizeごとに数字を飛ばす。0,100,200的な
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis = 1) #列番号を返す。
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
print((accuracy_cnt)/len(x))

a = np.array([[1,2,1,0],[5,2,1,0]])
np.argmax(a)#4 flattenしている
np.argmax(a,axis = 1) #行方向で比べる 今回は2,5が対象
np.argmax(a,axis = 0) #列方向で比べる。1,0,0,0 (5,2,1,0)となる。


#学習
#損失関数 #小さくするほど良い。
def mean_squared_error(y,t):  #yはsoftmax関数の出力層。t は教師データ
     y,t = np.array(y),np.array(t)
     return 0.5 * np.sum((y-t)**2)
#ex)
    #y = [0.1,0.2,0.6,0.1,0,0,0] #結果
    #t = [0,0,1,0,0,0,0] # 教師データ　True or False
    #mean_squared_error(y,t) #むっちゃ小さい

    #y1 = [0.1,0.2,0.6,0.1,0,0,0] #結果
    #t1 = [0,0,0,0,0,0,1] # 教師データ　True or False
    #mean_squared_error(y1,t1) #むっちゃ大きい

def cross_entropy_error(y,t):
    y,t = np.array(y),np.array(t)
    delta = 1e-7#10^-7
    return -np.sum(t*np.log(y + delta)) #deltaを入れることで、np.log(0) = infを防ぐ.

#ex)
    #y = [0.1,0.2,0.6,0.1,0,0,0] #結果
    #t = [0,0,1,0,0,0,0] # 教師データ　True or False
    #cross_entropy_error(y,t)
    #y1 = [0.1,0.2,0.6,0.1,0,0,0] #結果
    #t1 = [0,0,0,0,0,0,1]
    #cross_entropy_error(y1,t1)


#ミニバッチ学習 大量のデータの中から1部を抜き出して学習する.
#import sys, os,pickle
#sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
#import numpy as np
#from dataset.mnist import load_mnist
#from PIL import Image


#def img_show(img):
    #pil_img = Image.fromarray(np.uint8(img))
    #pil_img.show()

#(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

train_size = x_train.shape[0] #len(x_train)
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size) #10個ランダム
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
def cross_entropy_error2(y,t):
    if y.ndim == 1:
        y = y.reshape(1,y.size)
    t = t.reshape(1,t.size)
    batch_size = y.shape[0]
    delta = 1e-7#10^-7
    return -np.sum(t*np.log(y.T + delta))/batch_size #
cross_entropy_error2(x_batch,t_batch)

#確率的勾配降下法用の微分導入
#数値微分
def numerical_diff(f,x):
    h = 1e-4 #小さすぎるとアカーン　np.float32(1e-50) = 0.0とでてしまう。
    return (f(x+h)-f(x-f))/(2*h) # ダイレクトに行くと誤差が大きくなる。減らす方法として、中心差分を使う。
#偏微分
def numerical_gradient(f,x):
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

#勾配降下法
def gradient_descent(f,init_x,lr = 0.01,step_num = 100):#lr 学習率
       x = init_x.astype(float)
       for i in range(step_num) :
           grad = numerical_gradient(f,x)
           x -= lr * grad
       return x
def gradient_descent2(f,init_x,lr = 0.01,epsilon =1e-10):#lr 学習率
       x = init_x.astype(float)
       grad = numerical_gradient(f,x)
       while all(abs(lr*grad).flatten() > epsilon) :
           grad = numerical_gradient(f,x)
           x -= lr * grad
       return x

#ニューラルネットワークに対する勾配 dL/dW
#ex)from common.functions import softmax #sys.path.append(os.pardir)で親ディレクトリ参照可能
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)#ガウス分布で初期化

    def predict(self,x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss

net = simpleNet()
print(net.W)
x = np.array([0.6,0.9])
p = net.predict(x)
t = np.array([0,0,1]) #正解ラベル
net.loss(x,t)
def f(W):
    return net.loss(x,t)
dw = numerical_gradient(f,net.W)

#学習アルゴリズムの実装
#2層 machine-learning.py 参照
#テストデータで評価
#def エポック:すべてのデータを使い切ったら1エポック　ex)100バッチ10000個データなら100回回せば全部見れるので1エポック=100回。 上のプログラムに以下を加える。
train_loss_list = []
train_acc_list = []
test_acc_list = []
#1エポック当たりの繰り返し回数
iter_per_epoch = max(train_size/batch_size,1)
#1エポックごとに認識制度を計算
if i%iter_per_epoch ==0 :
    train_acc = network.accuracy(x_train,t_train)
    test_acc = network.accuracy(x_test,t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
