#4-1
import numpy as np
#乱数列生成
data = np.random.randn(10) # １行10列 作成 randn(2,3) # ２行３列
data = np.random.randn(2,3)　#data.shapeで確認できる
np.ones((3,4))#3行4列の1を作る 0 ならzeros
data * 10 # 各値を10倍
data + data #各値をたす

#4-1-1
data1 = [1,2,3,4]
arr1 = np.array(data1) # array環境になる
data2 = [[1,2,3,4],[1,2,3,4]]
arr2 = np.array(data2)
arr2.ndim　#次元
arr2.shape #(ry)
np.zeros((3,6)) #"3行6列にarrayで０を作る"
np.arange(15) #range(15)のarray作る
arr1.astype(np.int32)
arr1.astype(float)

#4-1-3
arr = np.array([[1.,2.,3.],[4.,5.,6.]])
arr2 = np.array([[6.,5,4.,],[3.,2.,1]])
arr2 > arr # 各要素のTrue Falseでリターンする
arr[5] #int etc
arr[5:8] #array([5,6,7])
arr[5:8] = 12 #12に変更
a = arr[5:8] # array([12,12,12])
a[1] = 12345 #arrにも影響が出る
a[:] # all に対して影響
arr2[1,2] = arr2[1][2]
a = arr[1].copy()#arrに影響なし
arr[:2][:1] #２行目まで抜き出され、1列以降が抜かれる

#4-1-5
names = np.array(["bob","mark","zero"])
data = np.random.randn(3,4)　#変更するときはnp.~.reshape() np.arange(9).reshape((3,3))
data[names == "bob"] #bob番目の行のデータ参照
data[~(names=="bob")]#真偽逆転
cond = names == "bob" #namesにTrueFalse
data[data < 0] = 0 #(ry)

#4-1-6
np.empty((8,4)) # ほぼzeroみたいな値が入る
arr[[1,2,3,4],[:0,2,1,3]]#1,2,3,4行目のデータを0,2,1,3列順で入れ替える
arr.T#転置
np.dot(arr,arr2)#内積
arr.transpose()#numpyのみ　指定がなければ転置
arr.swapaxes(1,2) #列と行を入れ替えて表示

#4-2 ユニバーサル関数
np.sqrt(arr)
np.exp(arr) #これらを単項ufunc(一つの引数を取る)

np.maximum(x,y) #x,yの各行の最大値を返す 2項func
a,b = np.modf(np.random.randn(7)*5) #小数と整数分ける
np.sqrt(arr,arr)#arrを直接変更

#4-3
points = np.arange(-5,5,0.01) #1000個の格子点
xs,ys = np.meshgrid(points,points) #(x,y)とし、https://deepage.net/features/numpy-meshgrid.htmlこのサイトを見よう
z = np.sqrt(xs**2 + ys**2)
#matplotlib.pyplot
import matplotlib.pyplot as plt
plt.imshow(z,cmap = plt.cm.gray);plt.colorbar() #後ほど

#4-3-1
arr = np.random.randn(4,4)
arr > 0 # true Falseでリターンする
np.where (arr>0,-2,2) #最初に条件文、次にtrue,falseの順
np.where(arr > 0,np.random.randn(4,4),np.random.randn(4,4)) #とかも可能

#4-3-2 統計関数
arr = np.random.randn(5,4)
arr.mean() # 全平均
np.mean(arr)#同じ
np.sum(arr) #arr.sum()
arr.mean(axis=1)#行ごとの平均
arr.sum(axis = 0)#列ごとの合計
arr.cumsum(axis = 0)#列ごとの累積和
arr.cumprod(axis= 1)#行ごとの累積積

#4-3-3　真偽値配列
(arr > 0).sum()
arr.sort()
arr.sort(axis = 1)
tile = np.random.randn(1000)
tile.sort()
tile[int(0.05 * len(tile))]
arr.unique()
sorted(set(arr))#上と同義
np.in1d(arr,[2,3,4]) #arrの各セルに対し、右の値があればTrue,else False

#4-4
arr = np.arange(10)
np.save("some_array",arr) #"some_array.npy"にセーブ
np.load("some_array.npy")
np.savez("some_array.npz",a = arr1, b= arr2)
arch = np.load("some_array.npz")
arch[b] #arr2

#4-5 行列計算
x = np.array([[1,2,3],[4,5,6])
y = np.array([6.,23.],[-1,7],[8,9]) #1行目に型が引っ張られる
x.dot(y) == np.dot(x,y) == x@y#内積
from numpy.linalg import inv,qr
x = np.random.randn(5,5)
mat = x.T.dot(x)
inv(mat)#逆行列

#4-6　疑似乱数生成
sample = np.random.normal(size = (4,4)) #正規分布に基づいた乱数生成 normal(loc = 平均,scale = 標準偏差,size = ())
sample2 = np.random.seed(1998)
rng = np.random.RandomState(1234)#seedを参照している
rng.randn(10) #randn は標準正規分布に依存　randintは範囲内での整数乱数　randは連続一様分布に従う乱数 binomialは二項分布に従う乱数

#4-7 randomwalk
import matplotlib.pyplot as plt
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0,1) else -1 #1がtrue 0 がfalse
    position += step
    walk.append(position)
plt.plot(walk[:100])
(np.abs(walk) >= 10).argmax() #argmaxは配列内の最大値のうち一番若いのを返す（今回は全部True or FalseなのでOK）
art = (np.abs(walk) >= 10).argmax(1)
art.mean() #一番最初に30に到達したタイムのmean

#4-7-1　多重ランダムwalk
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0,2,size = (nwalks,nsteps)) #numpyでは、0以上2未満
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1) # 行ごと
walks.max()
walks.min()
hits30 = (np.abs(walks) >= 30).any(1)#行ごと anyは一回でも感
hits30.sum()#20に到達した回数
