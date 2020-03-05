#10-1 groupby
df = ~
a = df.groupby(df["a"]).mean() #sqlのgroupby したものをaに入れる
b = df.groupby(df["a"],df["b"]).mean() #key1 の中でkey2でgroupby  unstack()すればdatarameっぽくなる
b = df.groupby(df["a"],df["b"]).size() #各groupのsize

#10-1-1　グループをまたいだ繰り返し
for name,group in df.groupby("key1"):
    print(name)
    print(group)
#name にkeyのgruop
#groupにkeyで分けたものの中身
for (key1,key2),group in df.groupby(["key1","key2"])
#(key1,key2) にkeyが反映
pieces = dict(list(df.groupby("key1")))
pieces["b"] #dataの中で好きなgroupからとってこれる

#10-1-2列や列の集合の選択
#もしgroupbyしてどこか一つだけ参照したい
df.groupby("key1")["data1"] == df["data1"].groupby("key1")

#10-1-3 ディクショナリやシリーズのグループ化
a = pd.DataFrame(~,columns = ["a","b","c","d","e"],index = ~)
mapping = {"a" : "red","b":"red","c":"red","d":"blue","e":"blue"}
~.groupby(mapping,axis = 1) #mappingでgroupbyする

#10-1-4 関数を使ったグループ化
~.groupby(len).sum()#lenでgroup それを和
~.groupby([len,key_list).sum()#lenとkey_listでgroup それを和

#10-1-5 インデックス階層によるグループ化
pd.MultiIndex.from_arrays([[],[]],names = [])
~.groupby(level = "~",axis = ).count()

#10-2 データの集約
df["~"].quantile(0.9) #分位点　

#自分で作成した関数で作られたテーブルの集約
#ex)
def peak_to_peak(arr):
    arr.max() - arr.min()
grouped.agg(peak_to_peak) #groupingして関数適用。
grouped.agg("mean") #meanで新行追加
grouped.describe()

#10-2-1 列に複数の関数を適用する
#aggの応用例
~.agg([("foo","mean"),("bar",np.std)])
grouped["a","b"].agg(["count","mean","max"])#aとbでgrouping 各々にagg内の関数で列作る。
grouped[~].agg([("a","mean"),("b","sum")])#groupingして、meanを列名aに、sumを列名bに　
grouped.agg({"tip":np.max,"size":"sum"})#50行目をまとめた感じ

#10-2-2 集約されたでデータを行インデックスなしで戻す。
.groupy(~,as_index = false) #indexなしでgroupby

#10-3 applyメソッド　一般的な分離-適用-結合方法
~apply(function,condition)#function関数のconditionをカンマで決める

#10-3-1グループキーの抑制
groupby(~,groupkey = False )#groupごとへのkeyがなくなる
#10-3-2 分位点とビン
#p-334
#10-3-3
#p-335
#復習
group_key = []
.groupby(group_key)#groupkeyをリストにしてそれをまとめる。

#10-3-4 ランダムサンプリングと順列
#ex)
suits = ["H","S","C","D"]
card_val = (list(range(1,11))+[10]*3)*4
base_names = ["A"] + list(range(2,11))+["J","K","Q"]
cards = []
for suit in ["H","S","C","D"]:
    cards.extend(str(num) + suit for num in base_names)
deck = pd.Series(card_val,index = cards)

#random sampling
def draw(deck,n=5):
    return deck.sample(n)

get_suit = lambda card : card[-1]
deck.groupby()

#相関を計算する
lambda x:x.corrwith(x[~])#xの特定の列とx[~]との相関
lambda x:x[~].corr(x[--]) #二つの特定の列相関

#10-3-6 線形回帰
import statsmodels.api as sm
def regress(a,b,c):
    Y = a
    X = b
    X["intercept"] = 1.
    result = sm.OLS(Y,X).fit()
    return result.params
#https://qiita.com/yubais/items/24f49df99b487fdc374d

#10-3 ピポットテーブルとクロス集計
~.pivot_table(index = ["a","b"])
~.pivot_table([~,~,=],index = [],columns =  )
.pivot_table([~,~],index = ~ ,columns = ~ ,margins = True )#columns の　分けなかった場合の平均が出る。
.pivot_table([~,~],index = ~ ,columns = ~ ,argfunc = len ,margins = True )#columns の　分けなかった場合の平均が出る。 argfuncで適用する関数選択
.pivot_table([~,~],index = ~ ,columns = ~ ,argfunc = len ,margins = True,fill_valie = )#nanを穴埋め #columns の　分けなかった場合の平均が出る。 argfuncで適用する関数選択

#10-4-1　クロス集計crosstab
pd.crosstab(data[~],data[~~],margins = True)#度数クロス表　marginで合計も出しやすく
#二段階もできる
pd.crosstab([a,b],c,margins = True)
