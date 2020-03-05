#14-1 データ分析の実例
# 短縮URL Bitlyにおける1.usa.govへの変換データ
import json
path = "~"
records = [json.loads(line) for line in open(path)]
#14-1-1 python標準機能でのタイムゾーン情報の集計
time_zone = [rec["tz"] for rec in records　(if "tz" in rec)] #error出る可能性あり　tzが入っていないデータもあるよね
#空白の部分を処理する
#ex)
def get_counts(sequence):
    counts = {}
    for x in sequence:
        if x in counts :
            counts[x] += 1
        else
            counts[x] = 1
    return counts

#上記と同じ
from collections import defaultdict

def get_counts2(sequence):
    counts = defaultdict(int) #初期値を0に設定
    for x in sequence:
        counts[x] += 1
    return counts

from collections import counter
counts = Counter(~)
counts.most_common(10) #上から10個

#14-1-2 pandas を使用したタイムゾーン情報の集計
import pandas as pd
tz_counts =  frame["tz"].value_counts() #tz count(*) from ~ group by index って感じ
clean_tz = framge["tz"].fillna("Missing")
clean_tz[clean_tz == ""] = "unknown" #空白のままだとグラフ化しにくい。文字にすることでカウント可能

#グラフ化
import seaborn as sns
subset = tz_counts[:10]
sns.barplot(y = subset.index, x= subset.values )
count_subset[]#dataframe 型
sns.barplot(x = "total", y="tz",hue = "os", data = count_subset) #hueが凡例


#14章で練習しよう
