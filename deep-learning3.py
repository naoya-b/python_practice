#誤差伝搬法
class MulLayer:
    def __init__(self): #初期化する
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y

        out = x * y

        return out
    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy
#ex)
apple = 100
apple_num = 2
tax = 1.1
#layer
mul_apple_Layer = MulLayer()
mul_tax_Layer = MulLayer()

#forward
apple_price = mul_apple_Layer.forward(apple,apple_num)
price = mul_tax_Layer.forward(apple_price,tax)

#backward
dprice = 1
dapple_price ,dtax = mul_tax_Layer.backward(dprice)
dapple,dapple_num = mul_apple_Layer.backward(dapple_price)

#加算レイヤー
class AddLayer:
    def __init__(self): #初期化しない
        pass
    def forward(self,x,y):
        out = x + y
        return out
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy

#ex)
#apple,apple_num,orange,orange_num,tax = 100,2,150,3,1.1
#layer
#mul_apple_Layer = MulLayer()
#mul_orange_Layer = MulLayer()
#add_apple_orange_Layer = AddLayer()
#mul_tax_Layer = MulLayer()

#forward
#apple_price = mul_apple_Layer.forward(apple,apple_num)
#orange_price = mul_orange_Layer.forward(orange,orange_num)
#all_price = add_apple_orange_Layer.forward(apple_price,orange_price)
#price = mul_tax_Layer.forward(all_price,tax)

#backward
#dprice = 1
#dall_price,dtax = mul_tax_Layer.backward(dprice)
#dapple_price,dorange_price = add_apple_orange_Layer.backward(dall_price)
#dorange,dorange = num = mul_orange_Layer.backward(dorange_price)
#dapple,dapple_num = mul_apple_Layer.backward(dapple_price)

#活性化関数レイヤ
class reLu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0 #xが0以下のとこだけ0になり、ほかは保たれる。
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx

#sigmoidレイヤ
class sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1 / (1+np.exp(-x))
        self.out = out

        return out
    def backward(self,dout):
        dx = dout*(1.0 - self.out) * self.out

        return dx

#affine/softmaxレイヤ
class Affine:
    def __init__(self,w,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b

        return out

    def backward(self,dout):
        dx = np.dot(dout,self.W.W)
        self.dW =np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0) #p151に理由記載
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #softmaxの出力
        self.t = None #教師データ

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y-self.t)/batch_size

        return dx

#SGD
class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr
    def update(self,params,grads):
        for key in params.keys():#dic型の(ry)
                params[key] -= self.lr * grads[key]

#Momentum p171
class Momentum:
    def __init__(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    def update(self,params,grads):
        if self.v is None :#初期化
            self.v = {}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            paramas[key] += self.v[key]

#AdaGrad
class AdaGrad:
    def __init__(self,lr = 0.01,epsilon = 1e-7):
        self.lr = lr
        self.h = None
        self.epsilon = epsilon

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + self.epsilon)
#過学習対策
#Dropout

class Dropout:
    def __init__(self,dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    def forward(self,x,train_flg = True):
        if train_flg:
            self.mask = np.random.randn(*x.shape) > self.dropout_ratio # https://dev.classmethod.jp/server-side/python/what-does-asterisk-mean-at-args/
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self,dout):
        return dout*self.mask

#ハイパーパラメータの検証
#ハイパーパラメータの検証のためのデータ作成
(x_train,t_train),(x_test,t_test) = load_mnist()

#訓練データをシャッフル　元データのままだと汎化性能が悪い。
x_train,t_train = shuffle_dataset(x_train,t_train)

#検証データの分割
validation_data = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_val = x_train[:validation_num] #検証データ　ハイパーパラメータ用
t_val = t_train[:validation_num] #
x_train = x_train[validation_num:] #テストデータ
t_train = t_train[validation_num:]

#ハイパーパラメータ最適化の実装
weight_decay = 10**np.random.uniform(-8,-4)
lr = 10 ** np.random.uniform(-6,-2)
