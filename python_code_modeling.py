#13-1 pandas とmodelのやり取りを行う
import pandas as pd
import numpy as np

#dataframe を　numpy 配列に直す
data = pd.DataFrame(~)
data.values #array(~)
df2 = pd.DataFrame(data.values,columns = [~])
data.loc[:,["a","b"]].values #範囲を指定
dummies = pd.get_dummies(data.category,prefix="~") #~列をdummieにする
data_with_dummies = data.drop("~",axis = 1).join(dummies)#dummy変数をjoin

#13-2 pastyを使ったモデルの記述
#Rに近いらしい
data = pd.DataFrame({"x0":~,"x1":~,"y":~})
import pasty
y,X = pasty.dmatrices("y~x0+x1",data)
y # y列のデータ
X #xo,x1の行列
np.asarray(y) # array化
np.asarray(X) # array化 dummy定数1が入る
y,X = pasty.dmatrices("y~x0+x1+0",data)
#+0を入れることで、切片校をなくす
coef,resid,_,_ = np.linalg.lstsq(X,y) #最小二乗法

#13-2-1 pasty式によるデータ変換
y,X = pasty.dmatrices("y~x0+np.log(np.abs(x1) + 1)",data)
y,X = pasty.dmatrices("y~standardize(x0)+center(x1)",data)
#standardize 標準化
#center 中心化 平均値を引く
new_data = pd.DataFrame(~)
new_X = pasty.build_design_matrices([X.design_info],new_data)#Xのデータをnew_dataに変更
y,X = pasty.dmatrices("y~I(x0+x1)",data) #I()にくくることでxo + x1の意味を足し算にできる

#13-2-2 カテゴリー型データとpasty
df = pd.DataFrame({
"key1":["a","b",~],
"key2" : [1,0,1,0,~]
"v2" : [1,2,3~]
})
y,X = pasty.dmatrices("v2 ~ key1",data)
X #key1 にダミー変数が与えられる。
y,X = pasty.dmatrices("v2 ~ key1+0",data) #key1[a],key1[b]にそれぞれダミー変数が入る。（aがあるところに1、bに0 bに１、aに0)
y,X = pasty.dmatrices("v2 ~ C(key2)",data)#key2をカテゴリーが他で読み込む
y,X = pasty.dmatrices("v2 ~ key1 + key2 + key1:key2",data) #key1:key2で&データを作る。

#statsmedels 入門
#13-3-1 線形モデルの推定
import statsmodels.api as sm
import statsmodels.formula.api as smf

#ex) ランダムデータから線形モデルを一個作る
def dnorm(mean,variance,size = 1):
    if isinstance(size,int):
        size = size,
    return mean + np.sqrt(variance)*np.random.randn(*size) #*変数で、変数をタプル化する https://pycarnival.com/one_asterisk/ *の意味がここに
#再現性のために乱数シード
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0,0.4,size = N),dnorm(0,0.6,size = N),dnorm(0,0.2,size = N)]
eps = dnorm(0,0.1,size = N)
beta = [0.1,0.3,0.5]

y = np.dot(X,beta) + eps
X_model = sm.add_constant(X)
#Xに切片１を加える。
model = sm.OLS(y,X) #線形回帰モデル
results = model.fit()
results.params
print(results.summary()) #いろいろ出る
results = smf.ols("y~a+b+c",data = data ).fit() #ひとまとめでできる
result.param
result.tvalues


#13-3-2 時系列モデルの推定
init_x = 4

import random
values = [init_x,init_x]
N = 1000

b0 = 0.8
b1 = -0.4
noise = dnorm(0,0.1,N)
for i in range(N):
    new_x = values[-1] * b0 + values[-2]*b1 + noise[i]
    values.append(new_x)

#AR(2)過程
MAXLAGS = 5
model = sm.tsa.AR(values)
results = models.fit(MAXLAGS) #ラグ指定

#13-4scikit-learn 入門
train = pd.read_txt("titanic.UTF")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

#交差検証 トレーニングデータを分割して、サンプル外のデータへの予測をシミュレートする
from sklearn.linear_model import LogisticRegressionCV
model_cv = LogisticRegressionCV(10) #精度を指定
model_CV.fit(X_train,y_train)

#自分で行い時
from sklearn.model_selection import cross_val_score
model = LogisticRegression(C = 10)
scores = cross_val_score(moedl,X_train,y_train,cv = 4)
