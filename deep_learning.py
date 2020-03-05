#前提
#cd ~ディレクトリ移動~
#python hungry.py #(hungry.pyを実行する)

#class
class Man : #image的には、np.array と似ている。 selfを変数に入れるのはデフォ。
    def __init__(self,name):#コンストラクタ、初期化
        self.name = name
        print("hello!")#メソッド1
    def hello(self):
        print("hello " + self.name + "!")
    def gb(self):#メソッド2
        print("gb"+self.name+"!")
m = Man("nao")#mにManのnaoを入れる。
m.hello()
m.gb()

#ブロードキャスト
a = np.array([[1,2],[3,4],[5,6]]) #2*3行列
b = np.array([1,2]) # 2*1行列
a[0,0] == 1 #要素へのアクセス
a.tolist() #リスト化
np.ndim(a)#nの次元
a.shape()#aの形
a*b #a[1]*b a[2]*b a[3]*bとなる。 例えば、2*3と2*2を同時にかけることはできない。
np.dot(a,b) #aとbの内積

a[a>3] # np.array([4,5,6])
a.flatten() #配列化

#画像の読み込み
from matplotlib.image import imread
img = imread("practice.png") #ディレクトリの変更を忘れずに 中にダイレクトにパスを入れてもよい
plt.imshow(img)
plt.show()


#実装
#1) EX ANDgate
def AND(X1,X2):
    w1 ,w2,b = 1,1,-1
    tmp = w1*X1 + w2*X2 +b
    if tmp <= 0:
        return 0
    else :
        return 1
#2) ANDGate nparray版
def neoAND(X1,X2):
    X = np.array([X1,X2])
    w = np.array([3,4])
    b = 3
    if all((X*w+b < 0).flatten():
        return 0
    else:
        return 1

 #3) step func
 def step_func(x): #xにはnp.array入る
     y = x > 1
     return y.astype(np.int)#True false等をintで出す
 #4)sigmoid
 def sigmoid(x):
     return 1/(1 + np.exp(-x))
#5)reLu関数
def reLu(x): #array入れると各値に対してみる。
    return np.maximum(0,x)

#3層ニューラルネットワーク実装
def nueral(X):
    W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    B1 = np.array([0.1,0.2,0.3])
    A1 = np.dot(X,W1) + B1 #第1層
    #2層目
    Z1 = sigmoid(A1)
    W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    B2 = np.array([0.1,0.2])
    A2 = np.dot(Z1,W2) + B2
    Z2 = sigmoid(A2)
    #3層目
    W3 = np.array([[0.1,0.4],[0.2,0.5]])
    B3 = np.array([0.1,0.2])
    A3 = np.dot(Z2,W3) + B3
    return A3
#まとめると
def init_network(): #辞書
    network = {}
    network["W1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["W2"] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network["W3"] = np.array([[0.1,0.4],[0.2,0.5]])
    network["B1"] = np.array([0.1,0.2,0.3])
    network["B2"] = np.array([0.1,0.2])
    network["B3"] = np.array([0.1,0.2])
    return network

def nuetal(network,x):
    W1,W2,W3 = network["W1"],network["W2"],#~
    #(以下略)
#nuetal(init_networl(),x)


#出力層
def softmax(a):
    exp_a = np.exp(a)
    sum_a = np.sum(exp_a) # これは、aが大きいと厳しい。
    y = exp_a/sum_a
    return y
#解決法
def newsoftmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 結果は変わらないけど、オーバーフロー対策
