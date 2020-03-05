#11-1 時系列データ
from datetime import datetime
now = datetime.now() #なんとなく
now.year(),now.month(),now.day() #tupleでreturn
delta = datetime(2017,8,11) - datetime(2017,4,12,5,15) # return delta.days delta.seconds
timedelta(~)#曜日のdelta指定　定数倍等可能

#11-1-1　文字列とdatetimeの変換
stamp = datetime(2011,1,3)
str(stamp) #redashでよく見る表記になる
a = stamp.strftime("%Y-%m-%d")#表記方法指定(date_format)
datetime.strptime(a,"%Y-%m-%d") #11行目と逆のことをする

from dateutil.parser import parse #pandas installでついてくる
parse("~") #時間（どんなものでもOK) datetime.datetimeで表示する
pd.to_datetime(~) # DatetimeIndex(~)
#p-350

#11-2 時系列の基本
from datetime import datetime
dates = [~,~,~]
ts = pd.Series(np.random.randn(~),index = dates)
ts[::2] #偶数番目抽出

#11-2-1 インデックス参照
stamp = ~.index[=]
ts[stamp] #indexのところの番号
pd.date_range("1/1/2000",periods = 1000) #1000日分
ts[datetime(2011,1,7):]
ts["1/6/2011":"1/11/2011"] #このような感じでindexを抽出できる。
ts.truncate(after = ~)#~以降のデータを取り除く
~.loc["5-2001"]

#12-2-2 重複したインデックスを持つ時系列
pd.TimeGrouper("5min") #groupbyのなかに入れることで5分ごとにbottleをつくる

#12-3-1 pipeメソッド
#ex)
a = f(df,arg1 = v1)
b = g(a,v2,arg = v3)
c = h(b,arg4 = v4)
result = (df.pipe(f,arg1 = v1).pipe(g,v2,arg = v3).pipe(h,arg4 = v4))
#pipe(関数、その関数に必要な変数)
