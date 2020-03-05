# 1 command prompt ディレクトリ
import os;os.getcwd() # 場所見る
cd 場所移動
cd .. 一つ上に戻る

#2.2.5
def -- 関数作る
ex)
def f(x,y,z):
return (x+y)/z
a=5
b=2
c=3 #a = 5;b=2;c=3と同義
result = f(a,b,c)
#別ファイルで
%run python_code_practice.py # 別ファイルでこの関数を動かせる
%paste #コピーしたものを一つのブロックとして貼り付け　間違えたらctrl+c

#matplotlib
#この二つを打つ
%matplotlib
import matplotlid.pyplot as plt
plt.show()
#で絵が出る

#2.3.1
a = [1,2,3]
b = a
a.append(4)
b # [1,2,3,4]

#import
ex import python_code_practice　as pcp #from python_code_practice import f と同義
result = python_code_practice.f(a,b,c) #pcp.f(a,b,c)

#変更可能　不可能object
a_list = ["foo",2,[4,5]]
a_list[2] = (3,4)
a_list #["foo",2,(3,4)] 変更可能
s_tuple = (1,2,(3,4))
s_tuple[2] ="four"
s_tuple # error

#2.3.2-2
c = """
~
~
""" #複数行にわたるときは"*3
a = "this is the song"
a[10] = f
a #error 文字列は不変
b = a.replace(song,sing)
b #予想通り
a #aに変化なし
list(a) #["t","h","i",(ry)]
a[:3] #"thi"

#浮動小数点 ex)
template = "{0:.2f}{1:s} are worth US${2:d}"
#{0:.2f} 1番目の引数を浮動小数点2桁
#{1:s} 2番目の引数を文字列
#{2:d} 3番目の引数を整数
template.format(4.556,"argentin pesos",1)
#ex format
print("a = {0},b = {1},c= {2}".format(1,2,3))

#2.3.2-7
def abc(a,b,c = none):
    result = a+b

    if c is not none: # cがデフォルト
        result = result *c

    return result

#日付
from datetime import datetime,date,time
dt = datetime(2011,1,2,3,4,5,6) #object
dt.day
dt.minute
dt.second #etc
dt.date()
dt.time()
dt.strftime("%m%d/%y")
datetime.strptime("20111031","%Y%m%d") # datetime.datetime(2011,10,31) 変更可能
dt.replace(minute = 0,second = 0)
delta = dt1 - dt2 #datetimetable
dt1 + delta #datetime table

#2.3.3-1 for if
if func < 0:
elif gun <0:
    return gun
elif x ==0:
    return 0
elif g < 0 :
    sequence = [1,2,5,6,7,8]
    for x in sequence:
        return x += x
        break
    for x in range(4):
        return x + 1
while x > 0:
    x = x - 100
    if x < 0:
        pass

range(10)
list(range(10))
list (range(5,0,-1)) # 5start 0end delta = -1

value = true content if condition else false-content

#~3章~
#tuple --固定長　変更不可能
tup = 4,5,6 #tuple made #(4,5,6)
tup = (4,5,6),(7,8) #tupの中にtupを入れるのもあり
tuple(list(range(4))) # tuple化
tuple("this") #文字もtuple化できる
tup[0] #"t" index を持つ
#tup + tup = (tup,tup)
#tup * 4 = tuptuptuptup
a,b,c = (1,2,3) # a = 1
z = a
a = b
b = z  # 入れ替え
b , a = a,b #で入れ替え　

a,*x = (1,2,3)
x #余りがlistになってくる

#~tuple function~
c = "this","is","is"
c.count(i) # 0

# list
list((1,2,3)) #[1,2,3]
#listは変更できるtuple
a = list(range(10))
a.append(1) # 最後に
a.insert(1,10)#a[1] = 10
a.pop(2) #a[2]番目を表示　a[2]が消える
a.remove(1) #aの中で最初にある1が消える
list + list = [list,list] #list.extend(list)のほうが望ましい

#sort
a = [1,2,7,5,6]
a.sort() #昇順
b = ["this","is","kame"]
b.sort(key = len) #lengthの昇順
import bisect
bisect.bisect(a,2)#sortの順を崩さない位置に2を入れる　その場所を返す
bisect.insort(a,5)#実際に入れる
b[1:2] = [1,2]
b #["this",1,2]
seq[::-1] # 反転する
seq[::2]#一つ飛ばし

#~シーケンスで使える関数~#
i = 0
for j in range(10):
    a[j] = j
    i + =1
#これと同義
mapping={}
for i,j in enumerate(list(range(10))):
    mapping[j] = i
mapping #{0:0,1:1,2:2 (ry)}

sorted("horse race") #listになってsortされたものがreturn

a = ["a","b","c"]
b = ["1","2",["3","4"]]
list(zip(a,b)) #  [('a', '1'), ('b', '2'), ('c', ['3', '4'])]
c = ["1","2"]
list(zip(a,b,c)) # : [('a', '1', '1'), ('b', '2', '2')]
 a = [("a","b"),("c","d")]
 c,d = list(zip(*a))
 c# ('a', 'c')

 list(reversed(["a","b","c"]) # a[::-1]

#~ディクショナリ~ mysqlのtableを作る感覚だよね
dict = {}
dt = {"a":"value",2:["this","is","pen"]}
dt[2] #["this" (ry)]
dt["this"] = "a" #辞書に追加
del dt["a"] #a:value が削除
a = dt.pop("this") # a
dt #"this":a が削除

#3-1-4-1
mapp = {}
for i,j in enumerate(list(("a","b","c"))):
    mapp[j] = i
for i,j in zip(("a","b","c"),("d","e","f")):
    mapp[i] = j

b ={}
words = ["apple","banana","atom"]
for a in words:
    letter = a[0]
    b.setdefault(letter,[]).append(a)
d = {}
d[tuple([1,2,3])] = 5

#3.1.5
set(list(range(10))) #setはunique かつ昇順 文字もOK
a = {1,2,3,4,5,5} #{}はuniqueにしてくれる
b = {7,8,9,9}
a.union(b) #unique かつ　昇順　
a | b
a.intersection(b) # a & b
c = a.copy() #(ry)
a & = b #s.t a= a & b
a = {tuple([1,2,3,4,4,3,2,1])} #{1,2,3,4,4,3,2,1}
set(tuple([1,2,3,4,4,3,2,1])) #{1,2,3,4}

#3.1.6
a = ["a","ab","abc"]
[x.upper() for x in a if len(x) > 2] #["ABC"] #{}でも可
set(map(len,a)) #map = applyみたいな
#ex)
a = input()
b,c = map(int,input().split())

#3-1-6-1
all_data = [["a","b"],["c","d"]]
for names in all_data:
    a = [r for r in names if r.count(names) >= 1] # a = [r for names in all_data for r in names if r.count("b") >= 1]

#3-2
def func(x,y,z):
    a = x + y + z
    if a > 0:
        return a
    elif a <= 0:
        return 0 #関数が終わるとaの値は削除
a = 0
def func(x,y,z):
    a = x + y + z
    if a > 0:
        return a
    elif a <= 0:
        return 0 #関数が終わるとaの値は削除

a = 0
def func(x,y,z):
    global a
    a = x + y + z
    if a > 0:
        return a
    elif a <= 0:
        return 0 #aは6になり、関数が終わっても大丈夫

# library re
b = ["  alabama","Georgia!","georgia","FLOriDA##"]
import re
def a(strings):
    result = []
    for a in strings:
        a = a.strip() #空白を消す
        a = re.sub("[!#?]"," ",a) #文字列の置換
        a = a.title() #文字列をタイトルにする
        result.append(a)
    return result

def remove(value):
    return re.sub("[!#?]","",value)
for x in map(remove,b): #作った関数でも可
     print(x)

#無名関数（ラムダ関数）
def func(x):
    return x**2
a_lambda = lambda x:x**2 #関数として定義 xがfunc(x)的な
a_lambda(2) # 4

#カリー化
def add_number(x,y):
    return x + y
add_five = lambda y: add_number(5,y) #関数を使って関数を定義

#print(end)
print(x,end="")#改行をなくす

#generater
#ex
sum(x**2 for x in rnage(10))
dict((i,i**2) for i in range(10))

#itertools
import itertools
first_name = lambda x:x[0]
names = ["Alan","Beta","Gamma","epsilon"]
for letter,names in itertools.groupby(names,first_name):
    print(letter,list(names))

#try - except
try --実行できるとき
except -- 実行できない時
finally --関係なく実行

#open --ほかのテキストファイルを参照
oath = "titanic2019_UTF.txt"
f = open(oath) #r環境で読み込まれる #read,seek,tellもある。read(10) 10字引っ張ってくる　seek(3) 3番目にスタート位置
for line in f:
    pass #逐次処理
lines = [x.rstrip() for x in open(oath)]
f.close() # ファイルを閉じれる
with open(path) as f:
    lines = [x.strip() for  in f]
