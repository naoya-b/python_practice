import pandas as pd
from pandas import Series,DataFrame
#5-1 pandasのデータ構造
#5-1-1 シリーズ
obj = pd.Series([4,7,-5,3]) # 0 4 || 1 7 || 2 -5 || 3 3
obj.values #array([4,7,-5,3])
obj.index #range(4)とおなじ RangeIndex(start = 0,stop = 4,step = 1) 左側
dataframe.columns #dataframe のcolumns

obj2 = pd.Series([4,7,-5,3],index=["a","b","3","d"]) #indexの変更が可能
obj2["d"]#3
obj2[["3","d","b"]] #3 -5 // d 3 //b 7
obj2[obj2 > 0]#結果は何となく
obj2 * 2 #数字部分が2倍に（indexは除く）
np.exp(obj2)#(ry)
"b" in obj2 #"True"
"e" in obj2 #False

sdata = {"ohio":35000,"texas":71000,"oregon":16000,"utah":5000}
states = ["texas","ohio",(ry)]
obj3 = pd.Series(sdata,index = states) #dictionaryがseriesに もし、indexの中に、sdataにないものがあった場合、return null
pd.isnull(obj3) #各tableにおいてreturn true false
pd.notnull(obj3)#上の逆
obj3.isnull()#OK
obj3 + obj4 #共通部分は計算され、ほかはnull
obj4.name = "population"
obj4.index.name = "state"

#5-1-2
data = {"state":["Ohio","ohio","ohio","Nevada","Nevada","Nevada"],
        "year" :[2000,2001,2002,2000,2001,2002],
        "pop" : [1,2,3,4,5,6,]
}
frame = pd.DataFrame(data) #tableの作成
frame.head(key)#key番目まで出力
frame2=pd.DataFrame(data,columns = ["year","state","pop"],index = []) #columnsで列順指定 indexでindex指定　なければnullが入る
frames["state"]
frame2.year
frame2.loc["three"] #３列目がseriesっぽく
frame2["debt"] = np.arange(6)#(ry)
frame2.debt = 15#(ry)
val = pd.Series([-1,-2,-3],index = [1,2,5])
frames["debt"] = val #frames の　debtの欄に合わせるように入る
frames["easten"] = frames2.state == "Ohio" #列の追加
del frames2.easten #列の削除

frame.T #転置
pdata = {"ohio":frames["ohio"][:2],
         "nevada" : frames["nevada"][:-1]} #ほかのテーブルから指定して作る

#5-1-3 index object
obj = pd.Series(range(3),index = ["a","b","c"])
index = obj.index
index #(ry)
index[1:]#index(["b","c"])
index[1] = "d" #indextableでの変更不可能
du.labels = pd.Index(["a","a","b"]) #同じのが入っても可能 setではuniqueになってしまう

#5-2 pandasの重要な機能
#5-2-1 再index付け
obj = pd.Series([4,8,-5,3],index = ["d","b","a","c"])
obj2 = obj.reindex(["a","b","c","d","e"]) #eではnone
obj2 = obj.reindex(["a","b","c","d","e"],method = "ffill") #noneのところに前方の値が穴埋め
obj2.reindex(columns = states,index = exampls)
#ex)
frame = pd.DataFrame(np.arange(9).reshape((3,3)),index = ["a","b","d"],columns = ["ohio","Texas","NY"])
frame.reindex(columns = ["Texas","NY","ABC"]) #ABC列がnoneに
frame.lot[["a","b","c","d"],["Texas","NY","ABC"]] # ABC列と"c"行がnoneに

#5-2-2 軸から要素削除
obj = pd.Series(np.arange(5.),index = ["a","b","c","d","e"])
new_obj = obj.drop(["c","d"]) #c,d行が消える
data = frame.copy()
data.index.name = "year";data.columns.name = "minutes"
data.drop([]) # 行削除
data.drop([],axis = 1 or "columns")#列から削除
data.drop([],inplace = True) #非可逆

#5-2-3　参照、選択、フィルタリング
data[data < 2]#none or return
data[data == 1] #dataが1のところだけ　ほかはnone
data[:2] #２行目まで
data[data["three"] > 5]#threeが5より大きいところすべてのtable return

#5-2-3-1 loc iloc
data.loc[[0,3],"state"] #行列指定
data.iloc[2,1] #２行１列してい
data.iloc[[2,3],[0,1]] #(ry)
data.iloc[][data.three > 5]

#5-2-4 整数のインデックス
#indexも整数だとser[-1]の時とかに何を参照するのかわかりにくい
ser[:1]#1行目まで
ser.loc[:1] #indexが1と書いてあるところまで
ser.iloc[:1]#1行目をseries型で

#5-2-5 算術　データ成型
s1 = pd.Series([7.3,-2.5,3.4,1.5],index = ["a","b","c","d"])#index = list((~))
s2 = pd.Series([-7.3,2.5,-3.4,-1.5,1],index = ["a","b","c","e","f"])
s1 + s2 #indexが同じところだけ四則計算
#dataframeの場合は同じ行列のところだけ足し算に　また、お互いにないところはnoneで補完されて作られる

#5-2-5-1 値の変換
df = pd.DataFrame(np.arange(12.).reshape((3,4)),columns = list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4,5)),columns = list("abcde"))
df2.loc[1,"b"] = np.nan
df2[1,"b"]#error
df.add(df2,fill_value = )#fill_valueのところで,df2のnoneの値を指定したものにする
1/df == df.rdiv(1)
df.reindex(columns = df2.columns,fill_value = 0)

#5-2-5-2
arr = np.arange(12).reshape((3,4))
arr[0]
arr-arr[0] # arr[0]が全部から引かれる
df -df.iloc[1] #上と同様

df.sub(frame,axis = 0 or "index")#マッチさせたい軸

#5-2-6
frame = pd.DataFrame(np.random.randn(4,3),columns = list("bde"),index = list("1234"))
np.abs(frame)
f = lambda x: x.max() - x.min()
frame.apply(f,axis = "index")
frame.apply(f,axis = "columns")
def f(x):
    return pd.Series([x.min(),x.max()],index = ["min","max"])
frame.apply(f,axis = "columns")

format = lambda x: "%.2f" % x
frame.applymap(format) #要素ごとに関数適用
frame["e"].applymap(format)

#5-2-7
s1.sort_index() #indexでソート
frame.sort_index(axis = ,ascending = FALSE)# axis =  でソート（昇順）する していなければaxis = 0 ascending = Falseで降順
s1.sort_values() #値でソート noneは最後に回される
frame.sort_values(by = ["a","b"] or ["a"] or etc) #複数でソート数場合、まず先頭でソート　次におんなじ値があれば2番目でソート
obj = pd.Series([7,-5.6,7,4,2.2,0,4.1])
obj.rank() #順位が出る。おんなじ数字があれば順位の平均が割り振られる　6位が3つあれば6.333etc
obj.rank(method = "first")#同じのがあれば観測された順に割り振る
obj.rank(ascending = false ,method = "max") #同じのがあれば、順位の値の大きいほうを代入する
frame.rank(axis = "")#行か列かでrank付け

#5-2-8 重複したラベルを持つ軸のインデックス
obj = pd.Series(range(5),index = []) #[]は重複を認めてる　int64
obj["a"] #index "a"なのが全部出る
frame["a"] # (ry)

#5-3 要約統計量の集計　計算
df = pd.DataFrame([[1,4],[2,np.nan],[0.75,np.nan],[np.nan,np.nan]],index = ["A","B","c","4"],columns = ["a","2"])
df.sum(axis = ) #axisを指定して和を出す naがあればそこは無視して足し算　ただしnan + nan = nan
df.mean(axis = ,skipna = False)#nanが出てくる行or列はnanで返す
df.idxmax(axis = ) #最大値を返すinde or columns を返す
df.cumcum()
df.describe(axis) #data数(excl none),mean,std（標本標準偏差）,min,25%,50%,75%,maxを返す #数値データの時

#5-3-1 相関と共分散
#command prompt で　conda install pandas-datareader
import pandas_datareader.data as web
all_data = {ticker:web.get_data_yahoo(ticker) for ticker in ["AAPL","IBM","MSFT","GOOG"]}
price = pd.DataFrame({ticker:data["Adj Close"] for ticker ,data in all_data.items()})
volume = pd.DataFrame({ticker:data["Volume"] for ticker ,data in all_data.items()})
returns = price.pct_change() #%変化を出す
returns["MSFT"].corr(returns["IBM"])#相関係数
returns["MSFT"].cov(returns["IBM"])#共分散 #returns.MSFT.cov(returns.IBM)
returns.corr() #相関係数全部出す
returns.cov()#分散共分散行列
returns.corrwith(returns.IBM)#IBMと全データの相関係数を出す
returns.covwith()#(ry)

#5-3-2 一意な値,頻度の確認
obj = pd.Series(["c","c","c","a","b","c","d"])
uniques = obj.unique()
obj.value_counts() # == pd.value_counts(obj.values,sort = false)
mask = obj.isin(["b","c"])
obj[mask]


#6 データの読み込み　書き出し　ファイル形式
read_csv
read_table
read_fwf # 区切り文字のないデータ
read_clipboard #クリップボードから読み込む
read_excel
read_hdf #pandas を用いて書き出したHDF5file
read_html
read_json #javascript object notation
read_msgpack
read_pickle
read_sas
read_sql
read_stata
read_feather #こんなのがあります

df = pd.read_csv("hubble.csv")
df = pd.read_table("hubble.csv")
df = pd.read_csv("hubble.csv",header = )#headr指定
df = pd.read_csv("hubble.csv",names = ["a","b","c"])#columns指定
df = pd.read_csv("hubble.csv",names = ["a","b","c"],index_col = "c" or ["c","b"])#columns指定 #c列をindexにする
a = list(open(csv txt))#list化して開く それをa
df = pd.read_csv("hubble.csv",names = ["a","b","c"],index_col = "c" or ["c","b"],skiprows = [0,2,3])#columns指定 #c列をindexにする 0,2,3行目のみ
df = pd.read_csv("hubble.csv",nrows = 5) #5行のみ読み込む

#6-1-2 テキスト形式でのデータ書き出し
df = pd.read_csv("hubble.csv")
df.to_csv("hubble_oud.csv") #コンマ区切りで,hubble_out.csvにdfを書き出し
df.to_csv("hubble_out2.csv",sep = "|") #|区切りでout2に書き出し
df.to_csv("hubble_out3.csv",na_rep = "NULL") #空白をnullでout3に書き出し
df.to_csv("hubble_out4.csv",index = False , header = False) #index ,headerをなくす
df.to_csv("hubble_out4.csv",index = False , columns = []) #index をなくす 指定した列だけ取り出す

dates = pd.date_range("1/1/2000",periods=7)
ts = pd.Series(np.arange(7),index = dates)
ts.to_csv("example.csv")

#6-1-3 区切り文字で区切られた形式を操作する
import csv
f = open("hubble.csv")
reader = csv.reader(f)
for line in reader :
    print(line)
with open("hubble.csv") as f:
    lines = list(csv.reader(f)) #リストとして読み込み
header , values = lines[0],lines[1:]
data_dict = {h:v,for h,v in zip(header,zip(*values))}
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar= '"'
    quoting = csv.QUOTE_MINIMAL #コンマを含むような必要最小限のデータのみを囲いたい csv.QUOTE_ALL すべてのデータを囲う csv.QUOTE_NONNUMERIC 非数値データをすべて囲う csv.QUOTE_NONE 囲いは使わない

reader = csv.reader(f,dialect = my_dialect) #上記でdelimater = ~とかやったのをまとめたもの

#6-1-4 json data
#6-1-5 XML HTML
conda install lxml
tables = pd.read_html()
len(table2)
#6-1-5-1 lxml.objectify を使ったXMLの読み込み
#6-2 バイナリデータ形式で効率よくデータを書き出す（computerノミが読めるデータ）
df = pd.read_csv("hubble.csv")
df.to_pickle("hubble_pickle") #pickledataで出力
pd.read_pickle("hubble_pickle")

#6-2-1 HDF5形式の使用
#6-2-2 excelfile 読み込み
a = pd.ExcelFile("trip_information.xlsx")
b = pd.read_excel(a,"result")#sheetを指定する
writer = pd.ExcelWriter("trip_information.xlsx")
a.to_excel(writer,"Sheet2")
writer.save()

#6-3 web APIを用いたデータの取得
import requests
url = "https://api.github.com/repos/pandas-dev/pandas/issues"
resp = requests.get(url)
data = resp.json()
data[0] #data={{},{}}的な
issues = pd.DataFrame(data,columns = ["number","title"])

#6-4データベースからデータの取得
import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20),b VARCHAR(20),
c  REAL,       d INTERGER
);"""
con = sqlite3.connect("mydata.sqlite") #sqliteファイル作成
con.execute(query)
con.commit()
data = [("Atlanta","Georgia",1.25,6),("Tallahassee","Florida",2.6,3),("Sacramento","California",1.7,5)]
stmt = "INSERT INTO test VALUES(?,?,?,?)"
con.executemany(stmt,data)
con.commit()
cursur = con.execute("select * from test")
fet = cursur.fetchall()
cursur.description #列名作る
pd.DataFrame(fet,columns = [x[0] for x in cursur.description])
import sqlalchemy as sqla
db = sqla.create_engine("sqlite:///mydata.sqlite")
pd.read_sql("select * from test",db)

#7 データのクリーニングと前処理
# 7-1 欠損値の取り扱い
string_data = pd.Series(["aardvark","artichoke",np.nan,"avocado"])
string_data isnull() #True FALSE
strind_data[0] = None #Noneにする

#7-1 欠損値削除
from numpy import nan as NA #np.nan
data = pd.Series([1,NA,3.5,NA,7])
data.dropna() == data[data.notnull()] #datanotnull()は真偽値返す

data2 = pd.DataFrame([1,NA],[NA,3],[4,5])
data2.dropna(axis = ) #axisdata　の　naを含む行を削除する
data.dropna(axis = ,how="all")#axisdata のすべてがNAのデータのみ削除
dataframe.dropna(thresh = ) #何列指定をしてその中のnadataの行をすべて削除する

#7-1-2 欠損値を穴埋めする
#fillnaは新しいobjectにして返す
df.fillna(1) #dfのnaを1に変更
df.fillna({1:0.5,2:0})#列ごとにnaの値を変更
df.fillna(0,inplace = True) #元のobjectを直接変更
df.fillna(method = "ffill",limit = 2) #連続した穴埋めを最大2回まで行う
df.fillna(data.mean()) #データの平均を返す

#7-2データの変形
#7-2-1重複の削除
data = pd.DataFrame({"k1":["one","two"]*3+["two"],"k2":[1,1,2,3,3,4,4,]})
data.duplicate() #行が自分の前までの行と重複しているか否か
data.drop_duplicate()#上記でTrue行を消す
data.drop_duplicate(["k1","k2"],keep = "last") #lastにすると重複あった場合下のほうが残る　k1で見た重複消した後にk2での重複消す

#7-2-2　関数やmapを用いたデータの変換
data = pd.DataFrame({"food":["bacon","pulled pork","bacon","Pastrami","corned beef","Bacon","pastrami","honey ham","nova lox"],"onces" : [4,3,12,6,7.5,8,3,5,6]})
meat_to_animal = {
"bacon":"pig",
"pulled pork": "pig",
"pastrami":"cow",
"corned beef" : "cow",
"honey ham" : "pig",
"nova lox" : "salmon"
}
lowecased = data["food"].str.lower() #すべて小文字に
data["animal"] = lowecased.map(meat_to_animal) #data["food"].map(lambda x : meat_to_animal[x.lower()])

#7-2-3 値の置き換え
data = pd.Series([1.,-999.,2,-999.,-100.,3.])
data.replace(-999,np.nan)
data.replace([-999,-1000],np.nan)
data.replace([-999,-1000],[np.nan,0]) #({-999:np.nan,-1000:0})

#7-2-4 軸のインデックスの名前を変更する
data = pd.DataFrame(np.arange(12).reshape((3,4)),index = ["ohio","colorado","New York"],columns = ["one","two","three","four"])
transform = lambda x: x[:4].upper()
data.index.map(transform)
data.index = data.index.map(transform)
data.rename(index = str.title,columns = str.upper)
data.rename(index = {"OHIO":"INDIAN"},columns = {"three","peekaboo"},implace = True) #特定のindex,columnsの変更 implaceで元データごといじる

#7-2-5 離散化とビニング
ages = [20,22,25,27,21,24,47,31,61,45,41,32]
bins = [18,25,35,60,100]
cats = pd.cut(ages,bins) #(18,25] (18,25],(18,25],(25,35] ...的な
cats.codes#codeが割り振られている
cats.categories #(]の種類
pd.value_counts(cats) #各binに含まれるデータをseriesっぽく返す
pd.cut(ages,[18,26,36,61,100],right = False)
group_names = ["Youth","youngadult","middleAged","Senior"]
pd.cut(ages,bins,labels = group_names)#(18,25] = Youth,(youngadult) =(25,35]
data = np.random.rand(20)
pd.cut(data,4,precision = 2) #４つの範囲で、小数点２桁の正確さ
pd.qcut(data,4)#4つの四分位範囲のビンに分割

#7-2-6 外れ値の検出と除去
data = pd.DataFrame(np.random.randn(1000,4))
data.describe()
col = data[2]
col[np.abs(col) > 3] #絶対値3以下を持ってくる
data[(np.abs(data) > 3).any(1)] #真偽値のデータフレーム
data[(np.abs(data) > 3)] = np.sign(data)* 3#signは正負の値に応じて1or-1を返す

#7-2-7　順列やランダムサンプリング
df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
sampler = np.random.permutation(5) #randomでsample数5
df.take(sampler) #ilocと似ているsamplerの順に並べる
df.sample(n=3) #randomに3行取り出して否復元
choices = pd.Series([5,-1,7,6,4])
draws = choices.sample(n=10,replace = True) #復元抽出でサンプル生成

#7-2-8 標識変数やダミー変数の計算
df = pd.DataFrame({"key":["b","b","a","c","a","d"],"data1" : range(6)})
pd.get_dummies(df["key"])　#unique keyにおいて各数字が出てきたかどうか(0,1)return
dummies = pd.get_dummies(df["key"],prefix = "naoya") #prefix naoya_a naoya_b 的になる
df_with_dummy = df[["data1"]].join(dummies) #next-time joinの使い方知ろう

#7-3 文字列操作
#7-3-1　文字列オブジェクトのメソッド
val = "a,b, guido"
val.split(",") #a,b,  guido
pieces = [x.strip() for x in val.split(",")] #a,b,guido
"::".join(pieces) #a::b::guido
val.find(~)#~が存在:1 存在しない:-1
val.index(~)#~が存在しなかったらerror

#7-3-2 正規表現
# mail-addressをいじるのはここらへん
import re
text = "foo  bar\t baz  \tqux"
re.split("\s+",text) #１文字以上の空白文字（タブ、スペース、改行文字で分割）
#regex = re.compile("\s+");regex.split(text) と同じ regex.findall(text)で正規表現にマッチした文字列参照
#pattern =r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
#7-3-3 文字列関数のベクトル化
data = { "Dave" : "dava@google.com","Steve": "steve@gmail.com","rob": "rob@gmail.com","Wes":np.nan}
data.str.contains("gmail")#TrueFalseでreturn
data.str.findall(pattern,flags=re.IGNORECASE) #ベクトル化
data.str[:5]#スライス後頭文字5つ

#8-1 データラングリング：連結、結合、変形
#8-1 階層型インデックス 複数のインデックスの階層を軸に持たせる
data = pd.Series(np.random.randn(9),index = [["a","a","a","b","b","b","c","c","c"],[1,2,3,1,2,3,1,2,3]])
#a  1    0.773019
#   2   -0.596355
#   3    0.561841
#b  1   -1.249209
#   2   -0.571827
#   3   -1.447280
#c  1   -1.817190
#   2    2.000805
#   3   -1.721805
data.index #index参照
data["b":"c"]#(ry)
data[:,2] #(ry)
data.unstack() #seriesだったものをdataframeっぽく
data.unstack().stack()#stackはdataframeっぽいものを → series
frame = pd.DataFrame(np.arange(12).reshape((4,3)),index = [["a","a","b","b"],[1,2,1,2]],columns = [["ohio","ohio","Colorado"],["Green","red","Green"]])
frame.index.names = ["key1","key2"]
frame.columns.names = ["col1","col2"] #(ry)

#8-1-1階層の順序変更やソート
frame.swaplevel("key1","key2") #key1 key2の値チェンジ
frame.sort_index(level = 1) #abab 1122 (key2の昇順) 上のswaplevelと組み合わせも可能

#8-1-2 階層ごとの要約統計量
frame.sum(level = "key2") #key2のuniqueで合計
frame.sum(level = "col2",axis = 1) #colorのuniqueで合計

#8-1-3 dataframeの列をindexに使う
frame.set_index(["c","d"],drop = True)#cとdをindexにする drop = Trueでindex行削除　falseで残る

#8-2 データセットの結合とマージ
#8-2-1 データフレームをデータベース風に結合する
df1 = pd.DataFrame({"key":["b","b","a","c","a","a","b"],"data1":range(7)})
df2 = pd.DataFrame({"key":["a","b","c"],"data2":range(3)})
pd.merge(df1,df2, on "key") #keyでmerge
#if df1 key1 df2 key2
pd.merge(df1,df2,left_on = "key1",right_on = "key2")
pd.merge(df1,df2,on = ~ ,how = "left") #right outer innerとかある　sqlっぽいよね

#8-2-2indexによるマージ
left1 = pd.DataFrame({"key":["a","b","a","b","a","b"],"value" = range(6)})
right1 = pd.DataFrame({"group_val" : [3.5,7],index = ["a","b"]})
pd.merge(left1,right1,leht_on = "key",right_index = True, how = ~)#Trueでindexをマージキーにできる

#8-2-3軸に沿った連結
arr = np.arange(12).reshape((3,4))
np.concateate([arr,arr],axis = 1) # 第１軸方向に二つを連結
s1 = pd.Series([0,1],index = ["a","b"])
s2 = pd.Series([2,3,4],index = ["c","d","e"])
s3 = pd.Series([5,6],index = ["f","g"])
pd.concat([s1,s2,s3]) #縦につなげる (行方向)
pd.concat([s1,s2,s3],axis = 1,join = "") #列方向につなげる
pd.concat([s1,s2,s3],axis = 1,join_axis =[["a","c","b","e"]]) #列方向につなげる　連結に使う軸指定
pd.concat([s1,s2,s3],keys = ["~","~","~"]) #にインデックスがつく
pd.concat([s1,s2,s3],axis = 1 , keys = ["~","~","~"]) #カラムがつく

#8-2-4重複のあるデータの結合
a = pd.Series([np.nan,2.5,0.0,3.5,4.5,np.nan],index = ["f","e","d","c","b","a"])
b = pd.Series([np.nan,2.5,0.0,3.5,4.5,np.nan],index = ["a","b","c","d","e","f"])

np.where(pd.isnull(a),b,a) #aが欠損地の場合bからアタイトル == a.conbine_first(b) dataframeでもできる

#8-3 変形とピポット操作
#8-3-1 階層インデックスによる変形
stack : データ内の各列を行へと回転
unstack : 各行を列へと回転させる。

data = pd.DataFrame(np.arange(6).reshape((2,3)),index = pd.Index(["Ohio","Colorado"],name = "state"),columns = pd.Index(["One","Two","Three"],name = "number"))
data.stack()
data.stack() .unstack()
#()の中に数字や列名入れるの可能
#dropna = False

#8-3-2 縦持ちフォーマットから横持ちフォーマットへ
dataFrame.pivot("data","item","value") #data - item のtableにする "value"書かないと階層構造のあるデータフレームになる
#8-3-3　横持フォーマットから縦持ちフォーマットへ
#出てきたときにみる
