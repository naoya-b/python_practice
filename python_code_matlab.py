#9-1 matplotlob api
%matplotlib
import matplotlib.pyplot as plt
plt.plot() #描画

#9-1-1サブプロット
fig = plt.figure()#空のプロットウィンドウ開く
ax1 = fig.add_subplot(2,2,1) #2*2の図を作れるようにする。それの1番目
plt.plot(~) #一番最後に作った場所に描画
plt.plot(,"k--")#黒色点線
ax1.hist(~) #ax1にヒストグラム
ax2.scatter()#ax2に点図
fig,axes = plt.subplots(2,3) #fig,axesそれぞれに2行3列のスペース作る P283参照

#9-1-1-1サブプロット、周りの空白調整
subplot_adjust(left = None,bottom = None,right = None,top = None,wspace = None,hspace = None )#幅設定　最後二つはサブプロット間の幅
#ex)
fig,axes=plt.subplots(2,2,sharex=True,sharey = True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500),bins = 50,color = "k",alpha = 0.5)
plt.subplots_adjust(wspace = 0 , hspace = 0)
plt.show() # 描画

#9-1-2　色、マーカー
#線種、色等はplt.plot?
~.plot(a,"g--") == ~.plot(b,linestyle = "--",color = "g")
.plot(marker = "o",label = "~") #黒点 label = ~で左上にlabel入る

#9-1-3メモリ　ラベル　凡例
#rと一緒　plt.xlim([0,10])#xの範囲は0~10

#9-1-3-1タイトル、軸ラベル、メモリ、メモリのラベル
~.set_xticks([0,250,500,750,1000])#メモリの何処にラベルを入れるか指定
.set_xtickslabels(["a","b","c","d","e"],rotation = 30,fontsize = "small") #rotationはlabelを回転させる　斜め　xtickslabelsはメモリのところに指定したラベル
.set_title("")

a = {
"title" : "~"
"xlabel" : "--"
}
~.set(**a)

#9-1-4 サブプロットへの注釈や描画
import matplotlib.dates as dates
from datetime import datetime #date導入

#日本語フォント設定
font_option = {"family":"TakaoGothic"}
plt.rc("font",**font_option)
#ex)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
data = pd.read_csv(,parse_dates = True)
spx = data["~"]
spx.plot(ax = ax,style = "~")
crisis_date = [
(datetime(2007,10,11),"a"),
(datetime(2008,3,12),"b")
]
for date,label in crisis_date: #(crisis_timeが()で入っている)
    ax.annotate(label,xy = (spx.asof(date)+75),xytest = (date,spx.asof(date)+225),arrowprops = dict(facecolor = "black",headwidth = 4,width = 2,headlength = 4),holizontalaligment = "left",verticalaligment = "top")

datefmt = dates.DateFormatter("%Y年%m月")
ax.xasis.set_major_formatter(datefmt)
ax.set_xlim(["1/1/2007","1/1/2011"])
ax.sex_ylim([600,1800])

#形
plt.Rectangle((0.2,0.75),0.4,0.13,color = ,alpha = )
plt.Circle((0.2,0.75),0.13,color = ,alpha = )
plt.Polygon()

#9-1-5　プロットのファイルへの保存
plt.savefig("figpath.svg",dpi = 400,bbox_inches = "~") #svgfileで保存,bbox_inchesで周りの空白を指定 400dpi(ドット数)で保存
#formatで使用するファイル形式の指定　名前もそれに合わせる

#9-1-6 matplotlibの設定
plt.rc("figure",figsize=(10,10)) #figure,axes,xtick,ytick,grid,legend等設定できる
font_option = {"family " : "","weight":"","size":""}
plt.rc("font",**font_option) #的な評価

#9-2 pandasとseabornのプロット関数
#9-2-1 折れ線グラフ
s = pd.Series(np.random.randn.(10).cumsum(),index=np.arange(0,100,10))
s.plot() #一本の折れ線グラフ

s = pd.Series(np.random.randn.(10,4).cumsum(0),columns = ["A","b","c","d"],index=np.arange(0,100,10))
s.plot() #4本の折れ線グラフ

#9-2-2 棒グラフ
fig,axes = plt.subplots(2,1) #figで大枠決め的な?
data = pd.Series(np.random.randn(16),index = list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0],color = "K",alpha = 0.7)
data.plot.barh(ax=axes[1],color = "K",alpha = 0.7)
df = pd.DataFrame(np.random.rad(6,4),index = ["a","b","c","d","e","f"],columns = pd.Index(["a","b","c","d"],name = "Genius"))
df.plot.bar(stacked = )#stacked= Trueなら積み上げる barhで横になる
pd.crosstab(a[""],a[""])#crosstabの使い方を後ほど
~.div(party_counts.sum(1),axis = 0)#各行の合計が１になるように正規化
import seaborn as sns
sns.barplot(x = "",y = "",data = ~ ,orient = "h") #棒グラフとともに95%信頼区間を表せる

#9-2-3 ヒストグラムと密度プロット
~.plot.hist(bins = ~) #50個の棒グラフ
~.plot.density()#滑らかの密度プロット
#ex)
comp1 = np.random.normal(0,1,size = 200)
comp2 = np.random.normal(10,2,size = 200) #平均　分散
values = pd.Series(np.concatenate([com1,com2]))
sns.distplot(values,bins = 100,color = "K")#密度推定と棒グラフ２種類出す

#9-2-4散布図
import seaborn as sns
sns.regplot("a","b",data = ~~) #散布図を作成して線形回帰により回帰直線を当てはめる
sns.pairplot(data = ~,diag_kind = "kde",plot_pws = {"alpha" : 0.2})#Rのpairs関数的なの

#9-2-5 ファセットグリッドとカテゴリーデータ
sns.factorplot(x = "~",y = "~",hue= "~",col = "~",kind = "~",data = ~) #col(columnsノミで場合分け)で場合分けしてグラフ出す　kind = bar 棒グラフ　box 箱ひげ図
sns.factorplot(x = "~",y = "~",row= "~",col = "~",kind = "~",data = ~) #time,col(row,columnにして場合分け)で場合分けしてグラフ出す
